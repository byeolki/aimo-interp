"""Test-time counterfactual check with a time manager and a self-probe fallback.

For each problem the target model answers the original twice and one rule-based variant per
family (rename, typos, distract), with a capped reasoning budget. A problem is robust when the
model keeps reproducing its own majority answer (aimo_interp.counterfactual). Problems that do
not fit in the time left are answered by the self-probe, computed first for every problem.

The 3600 s limit covers the whole prediction run, and calls arrive one model at a time, so the
time left is split evenly over the calls still expected for the Small Models track.
"""

import os
import sys
import time
from pathlib import Path

RUN_START = time.time()
SOLUTION_DIR = Path(__file__).resolve().parent
# The Codabench ingestion loads this file by path without adding its directory to sys.path.
if str(SOLUTION_DIR) not in sys.path:
    sys.path.insert(0, str(SOLUTION_DIR))

import torch  # noqa: E402

from aimo_interp.counterfactual import ORIGINAL, is_robust, plan_slots  # noqa: E402
from aimo_interp.features import load_model, release  # noqa: E402
from aimo_interp.models import resolve_model_id  # noqa: E402
from aimo_interp.perturb import FAMILIES  # noqa: E402
from aimo_interp.sampling import sample_answers  # noqa: E402
from aimo_interp.selfprobe import SelfProbe  # noqa: E402

TIME_LIMIT_S = float(os.environ.get("AIMO_TIME_LIMIT_S", 3600))
# Covers ingestion start-up before import, scoring overhead and the last chunk overrunning.
RESERVE_S = 420.0
EXPECTED_CALLS = 4
BUDGETS = (8192, 4096, 2048, 1024)
SEED = 20261101
# Seconds per decoding step for a full chunk before anything is measured; set from the
# benchmark in experiments/03-counterfactual and kept pessimistic.
PRIOR_STEP_SECONDS = float(os.environ.get("AIMO_PRIOR_STEP_SECONDS", 0.06))
# Fraction of free GPU memory handed to the KV cache of one generation chunk.
KV_MEMORY_FRACTION = 0.6

PROBE = SelfProbe(SOLUTION_DIR / "artifacts")
_loaded = None
_calls_done = 0


def _model(model_id: str):
    global _loaded
    if _loaded is None or _loaded.model_id != model_id:
        release(_loaded)
        _loaded = None
        _loaded = load_model(model_id)
    return _loaded


def _call_deadline() -> float:
    now = time.time()
    run_deadline = RUN_START + TIME_LIMIT_S - RESERVE_S
    remaining_calls = max(1, EXPECTED_CALLS - _calls_done)
    return now + max(0.0, run_deadline - now) / remaining_calls


def _sequences_per_chunk(loaded, budget: int) -> int:
    """How many sequences of ``budget`` new tokens fit in the free GPU memory."""
    if not torch.cuda.is_available():
        return 8
    config = loaded.model.config
    text_config = getattr(config, "text_config", config)
    layers = text_config.num_hidden_layers
    kv_heads = getattr(text_config, "num_key_value_heads", text_config.num_attention_heads)
    head_dim = getattr(text_config, "head_dim", None) or text_config.hidden_size // text_config.num_attention_heads
    bytes_per_token = layers * kv_heads * head_dim * 2 * 2
    free_bytes, _ = torch.cuda.mem_get_info()
    return max(5, int(free_bytes * KV_MEMORY_FRACTION // (bytes_per_token * (budget + 1024))))


def _plan_budget(loaded, n_problems: int, seconds_left: float, step_seconds: float) -> tuple[int, int] | None:
    """Largest budget whose chunks for all remaining problems fit in ``seconds_left``.

    Decoding is memory-bound, so a chunk takes about ``budget * step_seconds`` regardless of
    how many sequences it holds; bigger chunks are nearly free.
    """
    slots_per_problem = 2 + len(FAMILIES)
    for budget in BUDGETS:
        per_chunk = max(1, _sequences_per_chunk(loaded, budget) // slots_per_problem)
        chunks = -(-n_problems // per_chunk)
        if chunks * budget * step_seconds <= seconds_left:
            return budget, per_chunk
    # Not everything fits: spend what is left on as many problems as possible at the floor.
    budget = BUDGETS[-1]
    if budget * step_seconds <= seconds_left:
        return budget, max(1, _sequences_per_chunk(loaded, budget) // slots_per_problem)
    return None


def _counterfactual(model_id: str, problems: list[str], fallback: list[bool], deadline: float) -> tuple[list[bool], dict]:
    loaded = _model(model_id)
    predictions = list(fallback)
    stats = {"budgets": [], "counterfactual": 0}
    step_seconds = PRIOR_STEP_SECONDS
    index = 0
    while index < len(problems):
        plan = _plan_budget(loaded, len(problems) - index, deadline - time.time(), step_seconds)
        if plan is None:
            break
        budget, per_chunk = plan
        chunk = list(range(index, min(len(problems), index + per_chunk)))
        slots = plan_slots([problems[i] for i in chunk])
        started = time.time()
        results = sample_answers(loaded, [slot.text for slot in slots], budget, seed=SEED + index)
        # Keep the slower of prior and observation so one fast chunk cannot cause an overrun.
        step_seconds = max(step_seconds * 0.5, (time.time() - started) / budget)
        answers = {i: {ORIGINAL: [], **{f: [] for f in FAMILIES}} for i in range(len(chunk))}
        for slot, result in zip(slots, results):
            answers[slot.problem_index][slot.family].append(result.answer)
        for local, problem_index in enumerate(chunk):
            predictions[problem_index] = is_robust(answers[local])
        stats["counterfactual"] += len(chunk)
        stats["budgets"].append(budget)
        index = chunk[-1] + 1
    stats["fallback"] = len(problems) - stats["counterfactual"]
    return predictions, stats


def are_robust(model_id: str, reasoning_effort: str, problems: list[str]) -> list[bool]:
    """Predict robustness for each problem, preserving input order."""
    global _calls_done
    del reasoning_effort
    resolved = resolve_model_id(model_id)
    deadline = _call_deadline()
    _calls_done += 1
    try:
        fallback = PROBE.predict(resolved, problems, _model)
    except Exception as error:  # noqa: BLE001
        print(f"probe failed for {model_id}: {error!r}", file=sys.stderr)
        fallback = [PROBE.constant_guess for _ in problems]
    try:
        predictions, stats = _counterfactual(resolved, problems, fallback, deadline)
        print(f"{model_id}: {stats}, {time.time() - RUN_START:.0f}s elapsed", file=sys.stderr)
        return [bool(value) for value in predictions]
    except Exception as error:  # noqa: BLE001
        # Out-of-memory or generation errors must not invalidate the batch.
        print(f"counterfactual failed for {model_id}: {error!r}", file=sys.stderr)
        return fallback
