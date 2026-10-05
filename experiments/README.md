# Experiments

One directory per experiment, named `NN-short-name` (e.g. `01-layer-probe`). Each holds the
script, its config, and a `RESULTS.md` with the numbers and what was decided from them.
Raw outputs go to the ignored `runs/` directory.

Validation rules fixed before looking at data:

- Grouped cross-validation by problem (the same problem never in both train and validation).
- Report accuracy and balanced accuracy; tune thresholds on balanced accuracy.
- Never select layers or hyperparameters by leaderboard score.
