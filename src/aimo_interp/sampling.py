"""Batched sampling with a reasoning budget, using plain Transformers (no vLLM at test time).

A sample that hits the budget without finishing is forced to answer: the reasoning is closed
and the model greedily completes ``\\boxed{``. This mirrors the self-label generation, so the
offline numbers in experiments/02-self-labels carry over.
"""

from dataclasses import dataclass

import torch

from aimo_interp.answers import extract_integer
from aimo_interp.features import LoadedModel, format_problem

INSTRUCTION = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
FORCE_SUFFIX = "\n</think>\n\nTime is up. The final answer is \\boxed{"
FORCE_TOKENS = 16
# Organizers' FAQ sampling settings.
TEMPERATURE = 1.0
TOP_K = 40
TOP_P = 0.95


@dataclass(frozen=True)
class SampleResult:
    answer: int | None
    n_tokens: int
    forced: bool


def _input_device(model: torch.nn.Module) -> torch.device:
    return model.get_input_embeddings().weight.device


def _stop_token_ids(loaded: LoadedModel) -> set[int]:
    configured = loaded.model.generation_config.eos_token_id
    configured_ids = configured if isinstance(configured, list) else [configured]
    ids = {loaded.tokenizer.eos_token_id, loaded.tokenizer.pad_token_id, *configured_ids}
    return {token for token in ids if token is not None}


def _prompt(tokenizer: object, problem: str) -> str:
    return format_problem(tokenizer, problem + INSTRUCTION)


@torch.inference_mode()
def _generate(loaded: LoadedModel, prompts: list[str], max_new_tokens: int, sample: bool, seed: int) -> list[tuple[str, int, bool]]:
    tokenizer, model = loaded.tokenizer, loaded.model
    batch = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(_input_device(model))
    # Fixed seed per call keeps predictions identical across repeated evaluation runs.
    torch.manual_seed(seed)
    options = {"do_sample": True, "temperature": TEMPERATURE, "top_k": TOP_K, "top_p": TOP_P} if sample else {"do_sample": False}
    output = model.generate(**batch, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id, **options)
    new_tokens = output[:, batch["input_ids"].shape[1]:]
    eos_ids = _stop_token_ids(loaded)
    results = []
    for row in new_tokens:
        ids = row.tolist()
        finished = any(token in eos_ids for token in ids)
        length = next((i for i, token in enumerate(ids) if token in eos_ids), len(ids))
        results.append((tokenizer.decode(ids[:length], skip_special_tokens=True), length, finished))
    return results


def sample_answers(loaded: LoadedModel, problems: list[str], budget: int, seed: int) -> list[SampleResult]:
    """One sampled answer per entry of ``problems`` (repeat entries to get several samples)."""
    tokenizer = loaded.tokenizer
    prompts = [_prompt(tokenizer, problem) for problem in problems]
    generations = _generate(loaded, prompts, budget, sample=True, seed=seed)

    truncated = [i for i, (_, _, finished) in enumerate(generations) if not finished]
    forced_answers: dict[int, int | None] = {}
    if truncated:
        forced_prompts = [prompts[i] + generations[i][0] + FORCE_SUFFIX for i in truncated]
        completions = _generate(loaded, forced_prompts, FORCE_TOKENS, sample=False, seed=seed)
        forced_answers = {i: extract_integer("\\boxed{" + text) for i, (text, _, _) in zip(truncated, completions)}

    return [
        SampleResult(answer=forced_answers[i] if i in forced_answers else extract_integer(text), n_tokens=length, forced=i in forced_answers)
        for i, (text, length, _) in enumerate(generations)
    ]
