# 03 Test-time counterfactual check: results (2026-10-06)

## Method

`submissions/counterfactual`: at test time the target model answers the original problem twice
and one rule-based variant per family (rename, typos, distract) once, with a reasoning budget
chosen by a time manager (8k down to 1k tokens) and forced answers on truncation. The model's
own majority answer on the original is the pseudo-gold; robust iff no family loses more than
0.1 of it. Problems that do not fit in time fall back to the self-probe.

## Why it was tried

On our self-generated labels, re-scored on problems balanced per label (one robust and one
spurious model per problem), hidden-state probes stripped of model identity were at chance
(0.42-0.49; `../02-self-labels/calibration.txt`), and the apparent probe gains came from
model-identity priors (model prior alone 0.68). The counterfactual rule reached 0.64-0.66 on
held-out halves of the same samples (`../02-self-labels/counterfactual_rule.txt`).

## Results against organizer labels

| | 34 official train rows (offline, 5090) | Codabench val (14, balanced) |
|---|---|---|
| always-true | 23/34 = 0.68 (balanced 0.50) | 0.50 |
| layer-probe v1 / self-probe v1 | trained on these rows | 0.64 / 0.64 |
| **counterfactual v1** | **13/34 = 0.38 (balanced 0.48)**, predicted robust 9/34 | **0.50** |

Runtime on one RTX 5090: 3072 s for 34 cases with all four models, no fallbacks, no OOM after
the chunk/budget back-off fix (first run: Olmo OOM, 0.53 with the self-probe answering Olmo).

## Reading

The rule does not transfer to the organizers' labels. It calls most cases spurious (9/34
robust vs 70% robust in the labels). With 2 + 3 samples, one off answer flips the call, and
the organizers label with ~10 samples, 100k-token budgets and LLM-written perturbation
families that our rule-based variants do not reproduce. Agreement with our own labels came
from sharing those same choices.

## Decision

Keep **self-probe v1** (Codabench 964172, 0.643) as the final submission. The counterfactual
result goes into the report as a negative result: behavioural agreement under cheap rule-based
perturbations does not recover the official robustness labels.

GPU cost of this step: about $1.2 (RTX 5090, ~2.3 h, plus two ~$0.05/h CPU boxes).
