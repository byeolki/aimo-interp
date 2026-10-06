"""Per-model self-probe: each target model is judged from its own hidden state.

For a model with a fitted blend head (artifacts/self_heads.json), run that model once over
the problem prompts, average the mean-pooled hidden states of the head's layers, and apply
the head. Any other model id falls back to the shared layer-probe v1 head.
"""

import json
import sys
from pathlib import Path

SOLUTION_DIR = Path(__file__).resolve().parent
# The Codabench ingestion loads this file by path without adding its directory to sys.path.
if str(SOLUTION_DIR) not in sys.path:
    sys.path.insert(0, str(SOLUTION_DIR))

import numpy as np  # noqa: E402

from aimo_interp.blend import BlendHead  # noqa: E402
from aimo_interp.features import extract_hidden_states, load_model, release  # noqa: E402
from aimo_interp.models import resolve_model_id  # noqa: E402
from aimo_interp.probe import LinearProbe  # noqa: E402
from aimo_interp.spec import Spec  # noqa: E402

ARTIFACT_DIR = SOLUTION_DIR / "artifacts"
CONFIG = json.loads((ARTIFACT_DIR / "self_heads.json").read_text(encoding="utf-8"))
FALLBACK_DIR = ARTIFACT_DIR / CONFIG["fallback_dir"]
FALLBACK_SPEC = Spec.load(FALLBACK_DIR)

_loaded = None


def _model(model_id: str):
    global _loaded
    if _loaded is None or _loaded.model_id != model_id:
        release(_loaded)
        _loaded = None
        _loaded = load_model(model_id)
    return _loaded


def _predict_self(model_id: str, problems: list[str]) -> list[bool]:
    entry = CONFIG["heads"][model_id]
    head = BlendHead.load(ARTIFACT_DIR / entry["file"])
    pooled = extract_hidden_states(_model(model_id), problems, layers=[int(l) for l in head.layers])
    return head.predict(pooled[entry["pooling"]].mean(axis=0))


def _predict_fallback(model_id: str, problems: list[str]) -> list[bool]:
    head = FALLBACK_SPEC.head_for(model_id)
    pooled = extract_hidden_states(_model(head.encoder_id), problems, layers=[head.layer])
    features = pooled[head.pooling][0]
    if head.uses_model_onehot:
        features = np.hstack([features, np.tile(FALLBACK_SPEC.model_onehot(model_id), (len(problems), 1))])
    return LinearProbe.load(FALLBACK_DIR / head.probe_file).predict(features)


def are_robust(model_id: str, reasoning_effort: str, problems: list[str]) -> list[bool]:
    """Predict robustness for each problem, preserving input order."""
    del reasoning_effort
    resolved = resolve_model_id(model_id)
    try:
        if resolved in CONFIG["heads"]:
            return _predict_self(resolved, problems)
        return _predict_fallback(resolved, problems)
    except Exception as error:  # noqa: BLE001
        # A failed batch is scored as invalid; a constant guess is strictly better.
        print(f"self-probe fallback for {model_id}: {error!r}", file=sys.stderr)
        return [FALLBACK_SPEC.fallback_prediction for _ in problems]
