# 02 Self-generated labels and self-probes: results (2026-10-06)

## Data

- 150 integer-answer competition problems (`data/problems_small.jsonl`: AIME 111, AMC 40,
  others 30 minus overlap), none in the official train/val data.
- Per model: original x 8 samples; for problems solved at least once, one variant per family
  (rename, typos, distract; rule-based) x 6 samples. Sampling as in the organizers' FAQ
  (T=1.0, top-k 40, top-p 0.95).
- Reasoning capped at 8k tokens. At a 12k cap Qwen3.5-4B was truncated on 90% of samples, so
  a truncated sample is forced to answer (`</think> ... \boxed{`). Labels therefore describe
  robustness at a fixed 8k reasoning budget.
- Label: max over families of (base acc - family acc); robust <= 0.10, spurious >= 0.25.
  A model that never solves a problem is labeled robust (accuracy cannot drop).
- Models: Qwen3.5-4B, DeepSeek-R1-0528-Qwen3-8B, Skywork-OR1-Math-7B. Olmo-3-7B-Think was
  dropped: 640 tok/s on a 5090 (full-head KV cache), > 4 h for stage A alone.
- Bug found and fixed: fp8 KV cache made Skywork (Qwen2.5 base) loop on single tokens
  (non-truncated accuracy 0.12). Re-run with bf16 KV: 0.98. Discarded run kept in
  `runs/02-self-labels/discarded-fp8/`.

Result: 393 labeled generated rows + 25 official rows; **44 of 150 problems have labels that
differ across models** (117 rows), the subset that matches the per-problem balanced test set.

## Out-of-fold accuracy (problem-grouped 5-fold, 3 seeds)

| Method | mixed problems (117 rows) | official rows (25) |
|---|---|---|
| per-model majority | 0.54 | 0.68 |
| shared v1 (DeepSeek encoder + one-hot) | 0.58 | 0.68 |
| self-probe, direct, best single depth | 0.61 | 0.57 |
| two-step (predict own solve rate, then robustness) | 0.61-0.63 | 0.45-0.51 |
| **blend of both routes (chosen)** | **0.63** | **0.64** |
| sampled base accuracy (oracle-ish, uses 8 generations) | 0.74 | n/a |

Chosen config: mean pooling, layers at relative depth 0.5-0.7 averaged, ridge alpha 10000,
direct C 0.05. Neighbouring bands give 0.63-0.64, so it is not a single-config spike.

Other findings for the report:

- A ridge probe on the model's own hidden state predicts its own solve rate at r = 0.68-0.74
  (out-of-fold). Four sampled answers predict it at r = 0.48 (majority-answer agreement, no
  gold answer). One forward pass beats test-time sampling at a fraction of the cost, so
  test-time perturbation/sampling was not pursued.
- With only two models (Qwen + DeepSeek) the two-step route looked much stronger (0.72 on
  mixed rows); adding Skywork brought it to 0.62. Small-sample optimism is large here.
- Without the base-zero-as-robust labels only 56 mixed rows remain; the direct probe then
  reaches 0.65 mixed / 0.61 official. The assumption matters and the organizers' handling of
  never-solved cases is unknown.

## Submission

`submissions/self-probe` (ZIP sha256 114d4b2a6539): per-model blend heads for the three
models; Olmo and any other id use the shared v1 head. Offline contract test on the 34
official rows: 0 invalid, 38 s wall time including loading four checkpoints, identical across
two runs. Codabench ID 964172.

## Limitations

- 8k reasoning budget with forced answers vs the organizers' 100k tokens.
- Only rule-based perturbation families; no rephrase/domain/expert variants.
- 150 problems, 44 with mixed labels; the gap to the shared probe (+5 pts) is about one
  standard error.
- No Olmo self-labels.
