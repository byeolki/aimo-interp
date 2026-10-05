# 01 Layer probe

Question: which layer of which model carries a linearly readable robustness signal, and
does it beat surface difficulty features?

Data: `data/labels.jsonl` (171 rows: 34 official train-main-v2 rows over the four small
models, 137 augmented MATH rows for DeepSeek-R1-0528-Qwen3-8B).

Steps (GPU host):

```bash
scripts/gpu_setup.sh
tmux new -d -s exp01 "experiments/01-layer-probe/run.sh > runs/01-layer-probe/log.txt 2>&1"
```

Then pick encoder/pooling/layer from `runs/01-layer-probe/cv.csv` and export:

```bash
uv run scripts/export_probe.py --encoder <id> --pooling <last|mean> --layer <n>
uv run scripts/build.py layer-probe
```

Caveats to carry into the report:

- 34 official rows means about 8 points of standard error on official accuracy.
- Choosing the best layer on the same CV inflates its score.
- The organizers' robust/spurious criterion differs from the proposal paper and is not
  published, so augmented labels may not match the test labeling exactly.
