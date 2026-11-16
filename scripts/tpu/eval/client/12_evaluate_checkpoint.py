#!/usr/bin/env python
"""Evaluate a checkpoint on ALFWorld's held-out splits (data/alfworld/test.jsonl = seen, 140 games;
test_unseen.jsonl = unseen, 134 games), served by vLLM, with the full conversation as memory
(run_episode.py).

  python 12_evaluate_checkpoint.py --base-url http://localhost:8010/v1 --model step250 --split unseen \\
      --output out.unseen.jsonl --summary out.unseen.summary.json

Defaults: temperature 0.4, top_p 1.0, top_k -1, min_p 0.0, max_tokens 4096, 30 game steps, thinking off.
Run one job per chip and avg@4 with ../evaluate_checkpoint_avg4.sh.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from alfworld_agent_utils import _create_alfworld_env, _extract_task  # noqa: E402
from model_client import ChatModel  # noqa: E402
from run_episode import run_episode  # noqa: E402

# task files with absolute game paths (scripts/tpu/setup_node.sh)
_ALFWORLD_DATA_DIR = os.path.expanduser("~/alf-data/lookahead_opd_tasks")
DEFAULT_SEEN_JSONL = f"{_ALFWORLD_DATA_DIR}/test.jsonl"
DEFAULT_UNSEEN_JSONL = f"{_ALFWORLD_DATA_DIR}/test_unseen.jsonl"


def _load_jsonl_games(path: str, split_label: str) -> list:
    tasks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append({"game_file": json.loads(line)["game_file"], "split": split_label})
    return tasks


def _load_tasks(
    seen_jsonl: str,
    unseen_jsonl: str,
    split: str,
    task_start: Optional[int],
    task_end: Optional[int],
) -> list:
    sources = []
    if split in ("seen", "both"):
        sources.append((seen_jsonl, "seen"))
    if split in ("unseen", "both"):
        sources.append((unseen_jsonl, "unseen"))

    tasks = []
    for path, label in sources:
        games = _load_jsonl_games(path, label)
        if task_start is not None or task_end is not None:
            games = games[task_start:task_end]
        tasks.extend(games)
    return tasks


def _split_shards(tasks: list, n_workers: int) -> list:
    n_workers = max(1, min(n_workers, len(tasks)))
    shards = [[] for _ in range(n_workers)]
    for i, task in enumerate(tasks):
        shards[i % n_workers].append(task)
    return [s for s in shards if s]


def _worker(
    worker_id: int,
    tasks: list,
    base_url: str,
    api_key: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    top_k: int,
    min_p: float,
    enable_thinking: bool,
    max_env_steps: int,
    mock: bool,
    shard_output_path: str,
    keep_step_log: bool,
):
    t0 = time.time()
    print(f"[worker {worker_id}] processing {len(tasks)} games")

    model_client = ChatModel(
        base_url=base_url,
        api_key=api_key,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        enable_thinking=enable_thinking,
        mock=mock,
    )

    with open(shard_output_path, "w") as f:
        for i, task in enumerate(tasks):
            game_file = task["game_file"]
            env = None
            try:
                env = _create_alfworld_env(game_file)
                observation, info = env.reset()
                task_description = _extract_task(observation)

                result = run_episode(
                    env=env,
                    model_client=model_client,
                    task_description=task_description,
                    observation=observation,
                    info=info,
                    history=None,
                    start_step=0,
                    max_steps=max_env_steps,
                )

                record = {
                    "game_file": game_file,
                    "split": task["split"],
                    "task_description": task_description,
                    "actions": result["actions"],
                    "env_rounds": result["env_rounds"],
                    "final_reward": result["final_reward"],
                    "done": result["done"],
                    "success": result["done"],
                    "error": None,
                }
                if keep_step_log:
                    record["step_log"] = result["step_log"]
            except Exception as e:  # noqa: BLE001 - a single bad game must not
                # take down the rest of this worker's shard.
                print(
                    f"[worker {worker_id}] ERROR on game_file={game_file}: "
                    f"{type(e).__name__}: {e}",
                    file=sys.stderr,
                )
                record = {
                    "game_file": game_file,
                    "split": task["split"],
                    "task_description": None,
                    "actions": [],
                    "env_rounds": 0,
                    "final_reward": 0.0,
                    "done": False,
                    "success": False,
                    "error": f"{type(e).__name__}: {e}",
                }
                if keep_step_log:
                    record["step_log"] = []
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass

            f.write(json.dumps(record) + "\n")
            f.flush()

            if (i + 1) % 5 == 0 or (i + 1) == len(tasks):
                print(
                    f"[worker {worker_id}] {i + 1}/{len(tasks)} games done "
                    f"(last split={record['split']}, success={record['success']}, "
                    f"error={record['error']})"
                )

    print(f"[worker {worker_id}] finished shard in {time.time() - t0:.1f}s")


def _write_summary(records: list, summary_path: str, checkpoint_label: str):
    split_labels = sorted(set(r["split"] for r in records))
    per_split = {}
    for label in split_labels:
        split_records = [r for r in records if r["split"] == label]
        n = len(split_records)
        n_success = sum(1 for r in split_records if r["success"])
        success_rounds = [r["env_rounds"] for r in split_records if r["success"]]
        n_errors = sum(1 for r in split_records if r.get("error"))
        per_split[label] = {
            "n_tasks": n,
            "n_success": n_success,
            "success_rate": n_success / n if n else 0.0,
            "avg_rounds_on_success": (
                sum(success_rounds) / len(success_rounds) if success_rounds else None
            ),
            "n_errors": n_errors,
        }

    n_total = len(records)
    n_success_total = sum(1 for r in records if r["success"])
    summary = {
        "checkpoint": checkpoint_label,
        "n_tasks_total": n_total,
        "n_success_total": n_success_total,
        "success_rate_overall": n_success_total / n_total if n_total else 0.0,
        "per_split": per_split,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True, help="e.g. http://localhost:8010/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True, help="served-model-name on the vLLM server")
    parser.add_argument("--checkpoint-label", default=None,
                         help="Free-text label recorded in the summary JSON "
                              "(default: same as --model)")
    parser.add_argument("--seen-jsonl", default=DEFAULT_SEEN_JSONL,
                         help="140-task 'seen' held-out split (test.jsonl)")
    parser.add_argument("--unseen-jsonl", default=DEFAULT_UNSEEN_JSONL,
                         help="134-task OOD 'unseen' split (test_unseen.jsonl)")
    parser.add_argument("--split", choices=["seen", "unseen", "both"], default="unseen",
                         help="'both' pools seen + unseen (274 tasks)")
    parser.add_argument("--task-start", type=int, default=None,
                         help="0-indexed slice start into each selected split's game "
                              "list. Unset (default) = use every game in the split(s)")
    parser.add_argument("--task-end", type=int, default=None,
                         help="0-indexed slice end (exclusive), paired with --task-start")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-env-steps", type=int, default=30,
                         help="as in training")
    parser.add_argument("--temperature", type=float, default=0.4,
                         help="eval temperature (training samples at 1.0)")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--enable-thinking", action="store_true",
                         help="off by default, as in training")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--mock-model", action="store_true")
    parser.add_argument("--keep-step-log", action="store_true", default=True)
    parser.add_argument("--no-keep-step-log", dest="keep_step_log", action="store_false")
    args = parser.parse_args()

    tasks = _load_tasks(
        args.seen_jsonl, args.unseen_jsonl, args.split,
        args.task_start, args.task_end,
    )
    if not tasks:
        raise SystemExit("No tasks loaded -- check --seen-jsonl/--unseen-jsonl/--split.")

    shards = _split_shards(tasks, args.workers)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    shard_paths = [f"{args.output}.shard{i}.tmp" for i in range(len(shards))]

    split_counts = {}
    for t in tasks:
        split_counts[t["split"]] = split_counts.get(t["split"], 0) + 1
    counts_str = ", ".join(f"{k}={v}" for k, v in sorted(split_counts.items()))
    print(
        f"Evaluating model={args.model!r} on {len(tasks)} tasks ({counts_str}) "
        f"across {len(shards)} worker(s), max_env_steps={args.max_env_steps}, "
        f"temperature={args.temperature}, mock={args.mock_model}"
    )

    ctx = mp.get_context("spawn")
    procs = []
    for i, (shard, shard_path) in enumerate(zip(shards, shard_paths)):
        p = ctx.Process(
            target=_worker,
            kwargs=dict(
                worker_id=i,
                tasks=shard,
                base_url=args.base_url,
                api_key=args.api_key,
                model_name=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                enable_thinking=args.enable_thinking,
                max_env_steps=args.max_env_steps,
                mock=args.mock_model,
                shard_output_path=shard_path,
                keep_step_log=args.keep_step_log,
            ),
        )
        p.start()
        procs.append(p)

    failed = False
    for p in procs:
        p.join()
        if p.exitcode != 0:
            failed = True
            print(f"WARNING: worker pid={p.pid} exited with code {p.exitcode}", file=sys.stderr)

    all_records = []
    for shard_path in shard_paths:
        if not os.path.exists(shard_path):
            continue
        with open(shard_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_records.append(json.loads(line))
        os.remove(shard_path)

    all_records.sort(key=lambda r: (r["split"], r["game_file"]))
    with open(args.output, "w") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    checkpoint_label = args.checkpoint_label or args.model
    summary = _write_summary(all_records, args.summary, checkpoint_label)

    n_errors = sum(1 for r in all_records if r.get("error"))
    per_split_lines = [
        f"  {label}: {d['n_success']}/{d['n_tasks']} = {d['success_rate']:.1%}"
        + (f" (avg {d['avg_rounds_on_success']:.1f} steps on success)" if d["avg_rounds_on_success"] else "")
        for label, d in sorted(summary["per_split"].items())
    ]
    print(
        f"Done: {len(all_records)}/{len(tasks)} tasks written to {args.output}\n"
        + "\n".join(per_split_lines) + "\n"
        f"  Overall: {summary['n_success_total']}/{summary['n_tasks_total']} = "
        f"{summary['success_rate_overall']:.1%}"
        f"{f', {n_errors} task(s) errored' if n_errors else ''}\n"
        f"Summary written to {args.summary}"
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
