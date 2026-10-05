# AIMO Interpretability Challenge: Small Models track

Entry for the [NeurIPS 2026 AIMO Interpretability Challenge](https://aimo-interp.github.io/)
([Codabench](https://www.codabench.org/competitions/16180/)). The task is to predict, per
problem-model pair, whether the model solves the problem robustly or relies on a spurious pattern.

## Deadlines (KST)

| Date | What |
|---|---|
| 2026-11-01 21:55 | Final submission (Codabench phase end) |
| 2026-11-15 | Technical report + open-sourced code |
| 2026-11-15 to 11-30 | Peer review of up to 3 reports |

## Layout

```
submissions/<name>/   one directory per submitted method (solution.py at root)
src/aimo_interp/      shared code: hidden-state extraction, probe training
experiments/<id>/     one directory per experiment: script, config, result summary
scripts/              build.py (deterministic ZIP), score_local.sh (official harness)
third_party/          getting-started kit, pinned as a git submodule
docs/submissions.md   log of every upload: date, ZIP hash, score
report/               technical report
data/ runs/ dist/     ignored: datasets, caches, built ZIPs
```

## Usage

```bash
git submodule update --init
uv sync                                   # versions pinned to the evaluation image
uv run scripts/build.py always-true       # -> dist/always-true-small.zip (add --main for Main track)
scripts/score_local.sh dist/always-true-small.zip
```

The public val-sample (24 small-track cases) only checks the submission contract; it is not a
preview of leaderboard scores. Do not choose layers or thresholds by leaderboard score.

## Evaluation runtime

Offline, one RTX PRO 6000 (96 GB), 3600 s for the whole prediction run. Only the standard
library and the packages pinned in `pyproject.toml` exist at evaluation time.
