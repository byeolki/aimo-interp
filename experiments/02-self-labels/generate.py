#!/usr/bin/env python3
"""Sample answers with vLLM to build self-generated robustness labels for one model.

Stage A: every problem, original text, K_ORIGINAL samples.
Stage B: only problems the model solves at least once; each perturbation family gets
VARIANTS_PER_FAMILY variants with SAMPLES_PER_VARIANT samples each. A problem the model
never solves cannot lose accuracy, so stage B would be wasted on it.

Appends one JSON line per prompt to runs/02-self-labels/<model>.jsonl and resumes from it.
Sampling settings follow the organizers' FAQ except the generation cap (see MAX_TOKENS).

Usage: python generate.py --model Qwen/Qwen3.5-4B
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aimo_interp.answers import extract_integer  # noqa: E402
from aimo_interp.models import safe_model_id  # noqa: E402
from aimo_interp.perturb import FAMILIES, make_variant  # noqa: E402
from aimo_interp.problems import read_problems  # noqa: E402

PROBLEMS = ROOT / "data" / "problems.jsonl"
OUTPUT_DIR = ROOT / "runs" / "02-self-labels"
K_ORIGINAL = 8
VARIANTS_PER_FAMILY = 1
SAMPLES_PER_VARIANT = 6
# The organizers allow 100k tokens. 12k keeps the run cheap; truncated answers count as wrong,
# which lowers base accuracy on the hardest problems. Reported as a limitation.
MAX_MODEL_LEN = 12288
INSTRUCTION = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
# Large chunks keep the batch full; small ones idle the GPU on the longest tail sequences.
CHUNK = 120


def prompt_records(problems, stage: str, solved: set[str]) -> list[dict]:
    records = []
    for problem in problems:
        if stage == "A":
            records.append({"problem_id": problem.problem_id, "family": "original", "variant": 0,
                            "text": problem.problem, "answer": problem.answer, "n": K_ORIGINAL})
            continue
        if problem.problem_id not in solved:
            continue
        for family in FAMILIES:
            for variant in range(VARIANTS_PER_FAMILY):
                # Built-in hash() is salted per process; resumed runs must regenerate identical variants.
                seed = int(hashlib.sha256(f"{problem.problem_id}:{family}:{variant}".encode()).hexdigest()[:8], 16)
                text = make_variant(problem.problem, family, seed)
                records.append({"problem_id": problem.problem_id, "family": family, "variant": variant,
                                "text": text, "answer": problem.answer, "n": SAMPLES_PER_VARIANT})
    return records


def load_done(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run(llm, tokenizer, records: list[dict], output: Path) -> None:
    from vllm import SamplingParams

    for start in range(0, len(records), CHUNK):
        chunk = records[start : start + CHUNK]
        prompts = [
            tokenizer.apply_chat_template([{"role": "user", "content": r["text"] + INSTRUCTION}],
                                          tokenize=False, add_generation_prompt=True)
            for r in chunk
        ]
        params = [
            SamplingParams(n=r["n"], temperature=1.0, top_k=40, top_p=0.95,
                           max_tokens=MAX_MODEL_LEN - len(tokenizer(p).input_ids) - 8, seed=index)
            for index, (r, p) in enumerate(zip(chunk, prompts), start=start)
        ]
        started = time.time()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        n_tokens = 0
        with output.open("a", encoding="utf-8") as handle:
            for record, result in zip(chunk, outputs):
                answers = [extract_integer(choice.text) for choice in result.outputs]
                lengths = [len(choice.token_ids) for choice in result.outputs]
                n_tokens += sum(lengths)
                handle.write(json.dumps({
                    **{k: record[k] for k in ("problem_id", "family", "variant", "answer")},
                    "answers": answers,
                    "correct": [a == record["answer"] for a in answers],
                    "lengths": lengths,
                    "truncated": [choice.finish_reason == "length" for choice in result.outputs],
                }) + "\n")
        elapsed = time.time() - started
        print(f"{start + len(chunk)}/{len(records)} prompts, {n_tokens / elapsed:.0f} tok/s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=None, help="first N problems only (smoke test)")
    parser.add_argument("--problems", type=Path, default=PROBLEMS)
    args = parser.parse_args()

    from vllm import LLM

    problems = read_problems(args.problems)[: args.limit]
    output = OUTPUT_DIR / f"{safe_model_id(args.model)}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    import torch

    engine_options: dict = {"kv_cache_dtype": "auto"}
    if torch.cuda.get_device_capability(0)[0] >= 12:
        # The vast.ai image ships nvcc 12.8 and FlashInfer JIT refuses sm_120 below 12.9, so on
        # Blackwell every FlashInfer path (attention, Qwen3.5 GDN prefill, sampler) uses Triton.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        engine_options = {"kv_cache_dtype": "fp8", "attention_config": {"backend": "TRITON_ATTN"},
                          "additional_config": {"gdn_prefill_backend": "triton"}}
    llm = LLM(model=args.model, max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.92,
              enable_prefix_caching=True, disable_log_stats=False,
              # KV usage stays under 40% at the default 256 sequences on an 80 GB card.
              max_num_seqs=512, **engine_options)
    tokenizer = llm.get_tokenizer()

    for stage in ("A", "B"):
        done = load_done(output)
        done_keys = {(d["problem_id"], d["family"], d["variant"]) for d in done}
        solved = {d["problem_id"] for d in done if d["family"] == "original" and any(d["correct"])}
        todo = [r for r in prompt_records(problems, stage, solved)
                if (r["problem_id"], r["family"], r["variant"]) not in done_keys]
        print(f"stage {stage}: {len(todo)} prompts to run", flush=True)
        run(llm, tokenizer, todo, output)


if __name__ == "__main__":
    main()
