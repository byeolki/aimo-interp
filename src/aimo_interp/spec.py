"""Submission artifact layout shared by the exporter and the submission runtime.

``spec.json`` maps each target model (or ``"*"`` for any other id) to a head: which
encoder checkpoint to run, which layer and pooling to read, and which probe file to apply.
A shared head appends the target model's one-hot to the hidden state.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

SPEC_NAME = "spec.json"
WILDCARD = "*"


@dataclass(frozen=True)
class Head:
    encoder_id: str
    pooling: str
    layer: int
    probe_file: str
    uses_model_onehot: bool


@dataclass(frozen=True)
class Spec:
    model_order: list[str]
    heads: dict[str, Head]
    # Used when the encoder cannot run (e.g. alias ids in the public val-sample).
    fallback_prediction: bool = True
    notes: dict[str, str] = field(default_factory=dict)

    def head_for(self, model_id: str) -> Head | None:
        return self.heads.get(model_id, self.heads.get(WILDCARD))

    def model_onehot(self, model_id: str) -> list[float]:
        return [1.0 if model == model_id else 0.0 for model in self.model_order]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        payload = {**asdict(self), "heads": {key: asdict(head) for key, head in self.heads.items()}}
        (directory / SPEC_NAME).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "Spec":
        payload = json.loads((directory / SPEC_NAME).read_text(encoding="utf-8"))
        heads = {key: Head(**head) for key, head in payload.pop("heads").items()}
        return cls(heads=heads, **payload)
