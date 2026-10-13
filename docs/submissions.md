# Submission log

One row per Codabench upload. Hash is the first 12 hex chars of the ZIP's sha256 as printed by
`scripts/build.py`. Local score is on the public val-sample.

| Date (KST) | Track | Method | ZIP sha256 | Commit | Local acc | Codabench acc | Notes |
|---|---|---|---|---|---|---|---|
| 2026-10-05 22:40 | Small | always-true | 4a3718e38f91 | 34043a0 | 0.375 | 0.500 (7/14) | ID 962479. Pipeline check. First try returned HTTP 502 and was not counted |
| 2026-10-06 00:54 | Small | layer-probe v1 | a4f63f4e1af2 | b7d0bad | n/a (trained on official rows) | 0.643 (9/14) | ID 962741. Diagnostic: shared DeepSeek-8B L34 probe, expected near chance on per-problem balanced labels. Scored 2026-10-06 after a Codabench worker outage. Coverage 1.0, 0 invalid. Rank 13-25 of 88 (tied block at 0.64) |
| 2026-10-06 16:55 | Small | self-probe v1 | 114d4b2a6539 | c8502cc | CV mixed 0.63 / official 0.64 | 0.643 (9/14) | ID 964172. Same val score as layer-probe v1; 14 cases cannot separate them. Per-model blend heads (Qwen, DeepSeek, Skywork); Olmo falls back to shared v1. Offline contract test 0 invalid, 38 s |
| 2026-10-06 18:50 | Small | counterfactual v1 | 96cd8c05ecb0 | e7d2d85 | official 34 rows: 0.38 (balanced 0.48) | 0.500 (7/14) | ID 964391. Worse than self-probe v1; not used as final. Test-time check: 2 original + 1 per rule-based family, model's own majority answer as pseudo-gold; self-probe fallback under the time manager |
| 2026-10-12 20:46 | Small | self-probe v2 (leakage-free) | c4b06e490a46 | e933753 | CV mixed 0.53 / official 0.57 (102 problems) | cancelled | ID 980278 stuck in Running for 23 h on llmc-gpu-1 (logs stop after model loading), cancelled 2026-10-13. Same code as self-probe v1, heads retrained without AIME 2024-2026 per organizer leakage advice |
| 2026-10-13 | Small | self-probe v2 (resubmit) | c4b06e490a46 | 61e1793 | same as 980278 | pending | ID 984768. Resubmission of the stuck 980278 |
