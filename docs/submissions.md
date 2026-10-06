# Submission log

One row per Codabench upload. Hash is the first 12 hex chars of the ZIP's sha256 as printed by
`scripts/build.py`. Local score is on the public val-sample.

| Date (KST) | Track | Method | ZIP sha256 | Commit | Local acc | Codabench acc | Notes |
|---|---|---|---|---|---|---|---|
| 2026-10-05 22:40 | Small | always-true | 4a3718e38f91 | 34043a0 | 0.375 | 0.500 (7/14) | ID 962479. Pipeline check. First try returned HTTP 502 and was not counted |
| 2026-10-06 00:54 | Small | layer-probe v1 | a4f63f4e1af2 | b7d0bad | n/a (trained on official rows) | 0.643 (9/14) | ID 962741. Diagnostic: shared DeepSeek-8B L34 probe, expected near chance on per-problem balanced labels. Scored 2026-10-06 after a Codabench worker outage. Coverage 1.0, 0 invalid. Rank 13-25 of 88 (tied block at 0.64) |
| 2026-10-06 16:55 | Small | self-probe v1 | 114d4b2a6539 | c8502cc | CV mixed 0.63 / official 0.64 | pending | ID 964172. Per-model blend heads (Qwen, DeepSeek, Skywork); Olmo falls back to shared v1. Offline contract test 0 invalid, 38 s |
