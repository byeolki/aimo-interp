#!/usr/bin/env python3
"""Sample answers with vLLM to build self-generated robustness labels for one model.

Stage A: every problem, original text, K_ORIGINAL samples.
Stage B: only problems the model solves at least once; each perturbation family gets
VARIANTS_PER_FAMILY variants with SAMPLES_PER_VARIANT samples each. A problem the model
never solves cannot lose accuracy, so stage B would be wasted on it.

Appends one JSON line per prompt to runs/02-self-labels/<model>.jsonl and resumes from it.
Sampling settings follow the organizers' FAQ except the generation cap (see MAX_TOKENS).

Usage: python generate.py --model Qwen/Qwen3.5-4B
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aimo_interp.answers import extract_integer  # noqa: E402
from aimo_interp.models import safe_model_id  # noqa: E402
from aimo_interp.perturb import FAMILIES, make_variant  # noqa: E402
from aimo_interp.problems import read_problems  # noqa: E402

PROBLEMS = ROOT / "data" / "problems.jsonl"
OUTPUT_DIR = ROOT / "runs" / "02-self-labels"
K_ORIGINAL = 8
VARIANTS_PER_FAMILY = 1
SAMPLES_PER_VARIANT = 6
# The organizers allow 100k tokens. At a 12k cap Qwen3.5-4B was truncated on 90% of samples,
# which turns "unsolved" into "ran out of budget". So reasoning is capped at THINK_BUDGET and a
# truncated sample is forced to answer (the organizers' FAQ mentions the same thinking-budget
# trick). Labels therefore describe robustness at a fixed 8k reasoning budget.
THINK_BUDGET = 8192
FORCE_TOKENS = 24
FORCE_SUFFIX = "\n</think>\n\nTime is up. The final answer is \\boxed{"
MAX_MODEL_LEN = 10240
INSTRUCTION = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
# Large chunks keep the batch full; small ones idle the GPU on the longest tail sequences.
CHUNK = 120


def prompt_records(problems, stage: str, solved: set[str]) -> list[dict]:
    records = []
    for problem in problems:
        if stage == "A":
            records.append({"problem_id": problem.problem_id, "family": "original", "variant": 0,
                            "text": problem.problem, "answer": problem.answer, "n": K_ORIGINAL})
            continue
        if problem.problem_id not in solved:
            continue
        for family in FAMILIES:
            for variant in range(VARIANTS_PER_FAMILY):
                # Built-in hash() is salted per process; resumed runs must regenerate identical variants.
                seed = int(hashlib.sha256(f"{problem.problem_id}:{family}:{variant}".encode()).hexdigest()[:8], 16)
                text = make_variant(problem.problem, family, seed)
                records.append({"problem_id": problem.problem_id, "family": family, "variant": variant,
                                "text": text, "answer": problem.answer, "n": SAMPLES_PER_VARIANT})
    return records


def load_done(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def force_answers(llm, prompts: list[str], texts: list[str]) -> list[int | None]:
    """Close the reasoning and greedily read a boxed answer from a truncated sample."""
    from vllm import SamplingParams

    if not prompts:
        return []
    forced = [prompt + text + FORCE_SUFFIX for prompt, text in zip(prompts, texts)]
    params = SamplingParams(n=1, temperature=0.0, max_tokens=FORCE_TOKENS)
    outputs = llm.generate(forced, params, use_tqdm=False)
    return [extract_integer("\\boxed{" + result.outputs[0].text) for result in outputs]


def run(llm, tokenizer, records: list[dict], output: Path) -> None:
    from vllm import SamplingParams

    for start in range(0, len(records), CHUNK):
        chunk = records[start : start + CHUNK]
        prompts = [
            tokenizer.apply_chat_template([{"role": "user", "content": r["text"] + INSTRUCTION}],
                                          tokenize=False, add_generation_prompt=True)
            for r in chunk
        ]
        params = [
            SamplingParams(n=r["n"], temperature=1.0, top_k=40, top_p=0.95, max_tokens=THINK_BUDGET, seed=index)
            for index, r in enumerate(chunk, start=start)
        ]
        started = time.time()
        outputs = llm.generate(prompts, params, use_tqdm=False)

        truncated_slots = [(i, j) for i, result in enumerate(outputs)
                           for j, choice in enumerate(result.outputs) if choice.finish_reason == "length"]
        forced = force_answers(llm, [prompts[i] for i, _ in truncated_slots],
                               [outputs[i].outputs[j].text for i, j in truncated_slots])
        forced_by_slot = dict(zip(truncated_slots, forced))

        n_tokens = 0
        with output.open("a", encoding="utf-8") as handle:
            for i, (record, result) in enumerate(zip(chunk, outputs)):
                answers = [forced_by_slot.get((i, j), extract_integer(choice.text)) if choice.finish_reason == "length"
                           else extract_integer(choice.text) for j, choice in enumerate(result.outputs)]
                lengths = [len(choice.token_ids) for choice in result.outputs]
                n_tokens += sum(lengths)
                handle.write(json.dumps({
                    **{k: record[k] for k in ("problem_id", "family", "variant", "answer")},
                    "answers": answers,
                    "correct": [a == record["answer"] for a in answers],
                    "lengths": lengths,
                    "truncated": [choice.finish_reason == "length" for choice in result.outputs],
                }) + "\n")
        elapsed = time.time() - started
        print(f"{start + len(chunk)}/{len(records)} prompts, {n_tokens / elapsed:.0f} tok/s, "
              f"{len(truncated_slots)} forced", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=None, help="first N problems only (smoke test)")
    parser.add_argument("--problems", type=Path, default=PROBLEMS)
    args = parser.parse_args()

    from vllm import LLM

    problems = read_problems(args.problems)[: args.limit]
    output = OUTPUT_DIR / f"{safe_model_id(args.model)}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    import torch

    engine_options: dict = {"kv_cache_dtype": "auto"}
    if torch.cuda.get_device_capability(0)[0] >= 12:
        # The vast.ai image ships nvcc 12.8 and FlashInfer JIT refuses sm_120 below 12.9, so on
        # Blackwell every FlashInfer path (attention, Qwen3.5 GDN prefill, sampler) uses Triton.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        engine_options = {"kv_cache_dtype": "fp8", "attention_config": {"backend": "TRITON_ATTN"},
                          "additional_config": {"gdn_prefill_backend": "triton"}}
    llm = LLM(model=args.model, max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.92,
              enable_prefix_caching=True, disable_log_stats=False, **engine_options)
    tokenizer = llm.get_tokenizer()

    for stage in ("A", "B"):
        done = load_done(output)
        done_keys = {(d["problem_id"], d["family"], d["variant"]) for d in done}
        solved = {d["problem_id"] for d in done if d["family"] == "original" and any(d["correct"])}
        todo = [r for r in prompt_records(problems, stage, solved)
                if (r["problem_id"], r["family"], r["variant"]) not in done_keys]
        print(f"stage {stage}: {len(todo)} prompts to run", flush=True)
        run(llm, tokenizer, todo, output)


if __name__ == "__main__":
    main()
