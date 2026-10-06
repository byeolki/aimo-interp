"""Test-time counterfactual check: does the model keep its own answer under perturbation?

No gold answer is available at test time, so the model's majority answer on the original
problem stands in for it. The pseudo drop is the largest fall, over perturbation families,
in how often that answer is reproduced. A model that never produces a parseable answer cannot
drop and is called robust, matching how never-solved cases are labeled.

Offline (experiments/02-self-labels, held-out halves, problems balanced per label):
2 original samples + 1 per family reach 0.66 vs 0.42-0.49 for hidden-state probes.
"""

import collections
import hashlib
from dataclasses import dataclass

from aimo_interp.perturb import FAMILIES, make_variant

ORIGINAL_SAMPLES = 2
SAMPLES_PER_FAMILY = 1
ROBUST_MAX_PSEUDO_DROP = 0.1
ORIGINAL = "original"


@dataclass(frozen=True)
class Slot:
    problem_index: int
    family: str
    text: str


def variant_seed(problem: str, family: str) -> int:
    return int(hashlib.sha256(f"{family}:{problem}".encode()).hexdigest()[:8], 16)


def plan_slots(problems: list[str]) -> list[Slot]:
    """Every generation needed for ``problems``, original samples first per problem."""
    slots = []
    for index, problem in enumerate(problems):
        slots.extend(Slot(index, ORIGINAL, problem) for _ in range(ORIGINAL_SAMPLES))
        for family in FAMILIES:
            variant = make_variant(problem, family, variant_seed(problem, family))
            slots.extend(Slot(index, family, variant) for _ in range(SAMPLES_PER_FAMILY))
    return slots


def pseudo_drop(answers_by_family: dict[str, list[int | None]]) -> float:
    original = answers_by_family.get(ORIGINAL, [])
    counts = collections.Counter(answer for answer in original if answer is not None)
    if not counts:
        return -1.0
    majority, count = counts.most_common(1)[0]
    reproduced = count / len(original)
    drops = [
        reproduced - sum(answer == majority for answer in answers) / len(answers)
        for family, answers in answers_by_family.items()
        if family != ORIGINAL and answers
    ]
    return max(drops, default=0.0)


def is_robust(answers_by_family: dict[str, list[int | None]]) -> bool:
    return bool(pseudo_drop(answers_by_family) <= ROBUST_MAX_PSEUDO_DROP)
