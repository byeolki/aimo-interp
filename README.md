# AIMO Interpretability Challenge: Small Models track

Entry and technical report for the [NeurIPS 2026 AIMO Interpretability Challenge](https://aimo-interp.github.io/)
([Codabench](https://www.codabench.org/competitions/16180/)). The task is to predict, per
problem-model pair, whether the model solves the problem robustly or relies on a spurious pattern.

Report: [`report/report.pdf`](report/report.pdf), *Prompt Probes Read Solve Rate, Not Robustness:
Negative Results from the AIMO Interpretability Challenge (Small Models Track)*.

## Results in one table

| Method | Our balanced labels | Official 34 rows | Codabench val (14) |
|---|---|---|---|
| always-true | 0.50 | 0.68 (balanced 0.50) | 0.50 |
| shared prompt probe (`submissions/layer-probe`) | 0.58 (mixed problems) | CV up to 0.78 | 0.643 |
| self-probe blend (`submissions/self-probe`, **final**) | 0.63 mixed, 0.42-0.49 within-model | trained on these rows | 0.643 |
| test-time counterfactual (`submissions/counterfactual`) | 0.64-0.66 | 0.38 (balanced 0.48) | 0.50 |

Details and caveats: `experiments/01-layer-probe`, `02-self-labels`, `03-counterfactual` (each has a
`RESULTS.md`). Every upload is logged in `docs/submissions.md`.

## Layout

```
submissions/<name>/   one directory per submitted method (solution.py at root)
src/aimo_interp/      shared code: features, probes, blend heads, sampling, perturbations
experiments/<id>/     scripts, configs and RESULTS.md per experiment
data/                 problem pool, self-generated labels, raw sampled answers
report/               technical report (build_report.py -> report.html -> PDF)
scripts/              build.py (deterministic ZIP), score_local.sh, label/problem fetchers
third_party/          getting-started kit, pinned as a git submodule
```

## Reproduce

```bash
git clone --recursive https://github.com/byeolki/aimo-interp && cd aimo-interp
uv sync                                    # pinned to the evaluation image versions

# Final submission from the committed artifacts
uv run scripts/build.py self-probe         # -> dist/self-probe-small.zip (sha256 114d4b2a6539...)

# Rebuild the artifacts (GPU): hidden states, then the blend heads
uv run scripts/fetch_labels.py
uv run experiments/01-layer-probe/extract.py
uv run experiments/01-layer-probe/extract.py --problems data/problems_small.jsonl --output-dir runs/features-self
uv run experiments/02-self-labels/export_blend.py

# Regenerate the self labels (GPU, vLLM in a separate venv)
bash experiments/02-self-labels/setup_vllm.sh
bash experiments/02-self-labels/run_models.sh Qwen/Qwen3.5-4B deepseek-ai/DeepSeek-R1-0528-Qwen3-8B Skywork/Skywork-OR1-Math-7B
uv run experiments/02-self-labels/build_labels.py

# Offline check with the official ingestion code
scripts/score_local.sh dist/self-probe-small.zip
```

Notes: reasoning is capped at 8k tokens with a forced answer on truncation; Skywork-OR1-Math-7B
needs a bf16 KV cache in vLLM (fp8 KV makes it loop on single tokens). Total rented compute for
all experiments was about USD 10.

## Evaluation runtime

Offline, one RTX PRO 6000 (96 GB), 3600 s for the whole prediction run. Only the standard
library and the packages pinned in `pyproject.toml` exist at evaluation time.

## License

Code is released under the MIT License (`LICENSE`). Problem statements in `data/` come from public
competition datasets on Hugging Face (AI-MO, MathArena) and remain under their original terms;
the self-generated labels and sampled answers are released under the same MIT terms.
