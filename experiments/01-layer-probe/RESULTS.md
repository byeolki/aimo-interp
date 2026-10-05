# 01 Layer probe: results (2026-10-05)

Run on one RTX 5090 (vast.ai), torch 2.12.1+cu130, transformers 5.13.0, sklearn 1.8.0.
Full per-layer numbers: `cv.csv`. Out-of-fold, problem-grouped 5-fold CV, 3 seeds.
"official" = the 34 train-main-v2 rows across the four small models.

## Baselines (official rows)

| Config | Acc | Balanced acc |
|---|---|---|
| per-model majority | 0.58 | 0.51 |
| surface text features + model one-hot | 0.63 | 0.48 |

Surface difficulty (length, digits, LaTeX density, number size) carries no signal on the
official rows (0.63 on the augmented MATH rows).

## Probes, best layer per configuration (official rows)

| Config | Encoder | Pooling | Best layer (rel. depth) | Balanced acc |
|---|---|---|---|---|
| self (12 rows) | Qwen3.5-4B | last | 17 (0.53) | 0.90 |
| shared | DeepSeek-R1-0528-Qwen3-8B | last | 35 (0.97) | 0.78 |
| shared | Qwen3.5-4B | mean | 13 (0.41) | 0.76 |
| shared | Qwen3.5-4B | last | 12 (0.38) | 0.71 |
| shared | Skywork-OR1-Math-7B | mean | 18 (0.64) | 0.66 |
| shared | Olmo-3-7B-Think | mean / last | 9 / 12 | 0.65 |

Best-layer numbers are optimistic (selected on the same CV). The more honest read is the
plateau:

- shared DeepSeek-8B last token rises with depth: mean 0.62 over all layers, **0.72 over the
  top quarter (layers 28 to 36)**, with no single-layer spike.
- shared Qwen3.5-4B mean pool plateaus at 0.70 to 0.76 over layers 9 to 16.
- self Qwen3.5-4B is 0.81 to 0.90 over layers 15 to 20, but on 12 rows that is one or two
  rows of difference.

On the augmented MATH rows every probe stays near 0.55 to 0.65, so the official-row signal
does not come from fitting the augmented set.

## Decision

Submission `layer-probe` v1: one shared probe on DeepSeek-R1-0528-Qwen3-8B, last prompt
token, **layer 34** (centre of the top plateau rather than the single best layer), trained on
all 171 rows, model one-hot appended. Any model id falls through to this head.

Contract checks before upload: offline (`HF_HUB_OFFLINE=1`) run through the official
ingestion/scoring on the 34 official rows, 0 invalid, 6 s wall time including model load;
two separate processes give identical predictions; local and GPU builds give the same ZIP
hash.

## Open questions for the report

- Is the DeepSeek signal "this problem is hard for small models" (a property of the problem)
  or something about the target model? The shared probe cannot tell; a self-probe per model
  with more labels can. This needs generated labels (planned 10/12 onward).
- With 34 official rows the standard error on official accuracy is about 8 points; the
  difference between the top configurations is inside that.

## Caveat found after the run (Discord, 2026-10-05)

Organizers (announcements, 2026-09-15/16) found that top leaderboard solutions separated labels
by problem features and fell below chance on new problems. The validation set was rebuilt so
labels are **balanced per problem and per model**: each problem is robust for some models and
spurious for others. The test set is built the same way.

In our 34 official training rows, 9 problems have labels for more than one small model and
**none of them has mixed labels**. So the 0.72 CV score can come entirely from reading problem
difficulty, which the per-problem balance cancels out. The shared probe adds the model only as
a one-hot bias (no problem x model interaction), so on a per-problem balanced set it should be
expected near chance. Treat layer-probe v1 as a diagnostic, not a competitive entry.

Next: per-model self-probes on the target's own activations, and self-generated labels where the
same problem gets different labels across models.
