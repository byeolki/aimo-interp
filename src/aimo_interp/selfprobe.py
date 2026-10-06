"""Self-probe predictions from exported artifacts (see experiments/02-self-labels/export_blend.py).

Models with a blend head are judged from their own hidden states; any other model id uses the
shared layer-probe v1 head stored under ``fallback/``.
"""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from aimo_interp.blend import BlendHead
from aimo_interp.features import LoadedModel, extract_hidden_states
from aimo_interp.probe import LinearProbe
from aimo_interp.spec import Spec

ModelLoader = Callable[[str], LoadedModel]


class SelfProbe:
    def __init__(self, artifact_dir: Path):
        self.artifact_dir = artifact_dir
        self.config = json.loads((artifact_dir / "self_heads.json").read_text(encoding="utf-8"))
        self.fallback_dir = artifact_dir / self.config["fallback_dir"]
        self.fallback_spec = Spec.load(self.fallback_dir)

    @property
    def constant_guess(self) -> bool:
        return self.fallback_spec.fallback_prediction

    def predict(self, model_id: str, problems: list[str], load: ModelLoader) -> list[bool]:
        if model_id in self.config["heads"]:
            entry = self.config["heads"][model_id]
            head = BlendHead.load(self.artifact_dir / entry["file"])
            pooled = extract_hidden_states(load(model_id), problems, layers=[int(l) for l in head.layers])
            return head.predict(pooled[entry["pooling"]].mean(axis=0))
        shared = self.fallback_spec.head_for(model_id)
        pooled = extract_hidden_states(load(shared.encoder_id), problems, layers=[shared.layer])
        features = pooled[shared.pooling][0]
        if shared.uses_model_onehot:
            features = np.hstack([features, np.tile(self.fallback_spec.model_onehot(model_id), (len(problems), 1))])
        return LinearProbe.load(self.fallback_dir / shared.probe_file).predict(features)
