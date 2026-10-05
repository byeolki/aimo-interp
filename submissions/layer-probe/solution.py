"""Linear probe on one layer's hidden state of the prompt, one forward pass per problem.

Artifacts live in ``artifacts/`` (see ``aimo_interp.spec``). The shared ``aimo_interp``
package is copied next to this file by ``scripts/build.py``.
"""

import sys
from pathlib import Path

SOLUTION_DIR = Path(__file__).resolve().parent
# The Codabench ingestion loads this file by path without adding its directory to sys.path.
if str(SOLUTION_DIR) not in sys.path:
    sys.path.insert(0, str(SOLUTION_DIR))

import numpy as np  # noqa: E402

from aimo_interp.features import extract_hidden_states, load_model, release  # noqa: E402
from aimo_interp.models import resolve_model_id  # noqa: E402
from aimo_interp.probe import LinearProbe  # noqa: E402
from aimo_interp.spec import Spec  # noqa: E402

ARTIFACT_DIR = SOLUTION_DIR / "artifacts"
SPEC = Spec.load(ARTIFACT_DIR)

# Calls arrive one model at a time; keeping the last encoder avoids reloading it when
# several targets share one encoder.
_loaded_encoder = None


def _encoder(encoder_id: str):
    global _loaded_encoder
    if _loaded_encoder is None or _loaded_encoder.model_id != encoder_id:
        release(_loaded_encoder)
        _loaded_encoder = None
        _loaded_encoder = load_model(encoder_id)
    return _loaded_encoder


def _predict(model_id: str, problems: list[str]) -> list[bool]:
    head = SPEC.head_for(model_id)
    if head is None:
        return [SPEC.fallback_prediction for _ in problems]
    pooled = extract_hidden_states(_encoder(head.encoder_id), problems, layers=[head.layer])
    features = pooled[head.pooling][0]
    if head.uses_model_onehot:
        onehot = np.tile(SPEC.model_onehot(model_id), (len(problems), 1))
        features = np.hstack([features, onehot])
    return LinearProbe.load(ARTIFACT_DIR / head.probe_file).predict(features)


def are_robust(model_id: str, reasoning_effort: str, problems: list[str]) -> list[bool]:
    """Predict robustness for each problem, preserving input order."""
    del reasoning_effort
    try:
        return _predict(resolve_model_id(model_id), problems)
    except Exception as error:  # noqa: BLE001
        # A failed batch is scored as invalid; a constant guess is strictly better.
        print(f"layer-probe fallback for {model_id}: {error!r}", file=sys.stderr)
        return [SPEC.fallback_prediction for _ in problems]
