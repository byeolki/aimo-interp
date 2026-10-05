"""Model identifiers used by the Small Models track and the public datasets."""

import re

SMALL_TRACK_MODELS: tuple[str, ...] = (
    "Qwen/Qwen3.5-4B",
    "Skywork/Skywork-OR1-Math-7B",
    "allenai/Olmo-3-7B-Think",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
)

# Public sample datasets use Codabench aliases; the starter kit documents this mapping.
# Competition-phase calls always pass the exact checkpoint ids above.
MODEL_ALIASES: dict[str, str] = {
    "qwen3-8b:low": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
}


def resolve_model_id(model_id: str) -> str:
    """Map a dataset alias to its Hugging Face checkpoint id; unknown ids pass through."""
    return MODEL_ALIASES.get(model_id, model_id)


def safe_model_id(model_id: str) -> str:
    """Return a filesystem-safe token for a model id, e.g. ``Qwen_Qwen3.5-4B``."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")
