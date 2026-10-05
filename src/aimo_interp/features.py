"""Hidden-state extraction: one forward pass over the formatted prompt, no generation.

Used both for offline probe training and inside the submission, so prompt formatting
and pooling stay identical between the two.
"""

from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

POOLINGS: tuple[str, ...] = ("last", "mean")
MAX_PROMPT_TOKENS = 2048
DEFAULT_BATCH_SIZE = 8


@dataclass
class LoadedModel:
    model_id: str
    tokenizer: object
    model: torch.nn.Module


def load_model(model_id: str, local_files_only: bool = True) -> LoadedModel:
    """Load a checkpoint in bf16 on the single visible GPU (fp32 on CPU)."""
    has_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if has_cuda else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    # Left padding keeps the final prompt token at index -1 for every row in a batch.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto" if has_cuda else None,
        local_files_only=local_files_only,
    )
    model.eval()
    return LoadedModel(model_id=model_id, tokenizer=tokenizer, model=model)


def format_problem(tokenizer: object, problem: str) -> str:
    """User-only chat turn with the generation prompt, i.e. the state right before the
    model starts reasoning. Falls back to raw text for tokenizers without a template."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": problem}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return problem


def _input_device(model: torch.nn.Module) -> torch.device:
    return model.get_input_embeddings().weight.device


@torch.inference_mode()
def extract_hidden_states(
    loaded: LoadedModel,
    problems: list[str],
    layers: list[int] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, np.ndarray]:
    """Return ``{pooling: array[n_layers, n_problems, hidden]}`` in float32.

    Layer index 0 is the embedding output, following ``output_hidden_states``. ``layers``
    selects a subset (in the given order); ``None`` keeps all of them.
    """
    tokenizer, model = loaded.tokenizer, loaded.model
    prompts = [format_problem(tokenizer, problem) for problem in problems]
    pooled: dict[str, list[np.ndarray]] = {name: [] for name in POOLINGS}

    for start in range(0, len(prompts), batch_size):
        batch = tokenizer(
            prompts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_PROMPT_TOKENS,
            # Chat templates already contain BOS where the model expects it.
            add_special_tokens=False,
        ).to(_input_device(model))
        output = model(**batch, output_hidden_states=True, use_cache=False, return_dict=True)
        hidden = output.hidden_states
        selected = range(len(hidden)) if layers is None else layers
        stacked = torch.stack([hidden[index] for index in selected]).float()

        mask = batch["attention_mask"].to(stacked.device, dtype=stacked.dtype)[None, :, :, None]
        mean = (stacked * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1.0)
        pooled["last"].append(stacked[:, :, -1, :].cpu().numpy())
        pooled["mean"].append(mean.cpu().numpy())

    return {name: np.concatenate(chunks, axis=1) for name, chunks in pooled.items()}


def release(loaded: LoadedModel | None) -> None:
    """Free GPU memory before loading the next checkpoint."""
    if loaded is None:
        return
    del loaded.model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
