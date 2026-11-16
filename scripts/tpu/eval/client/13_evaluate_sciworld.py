#!/usr/bin/env python
"""Evaluate a checkpoint on ScienceWorld's test split (data/scienceworld/test.jsonl: 13 task types held out
from training, 1308 games), served by vLLM, with the rollout LookAheadOPD trains on: DASH-OPD's prompts
(arXiv 2607.29078, Appendix L) with TCOD's action list, and the full conversation as memory, built by the
same functions as the training workflow (lookahead_opd/scienceworld_utils.py).

Per game: score = best score / 100 (the reward), success = the task completed (score 100). A game whose
conversation outgrows the server's context (40960 tokens incl. max_tokens) ends at that turn with its best
score so far (context_overflow; reported as n_context_overflow, not as an error). The summary reports both:
mean score and success rate. Defaults: temperature 0.4, max_tokens 4096, 30 game steps, thinking off.

  python 13_evaluate_sciworld.py --base-url http://localhost:8081/v1 --model step250 \\
      --tasks-jsonl ~/LookAheadOPD/data/scienceworld/test.jsonl --output out.jsonl --summary out.summary.json

Run one job per chip and avg@4 with ../evaluate_sciworld_avg4.sh.
"""

import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from typing import List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(_THIS_DIR, "../../../..")))  # the repo

from model_client import ChatModel  # noqa: E402
from trinity.common.workflows.envs.lookahead_opd import scienceworld_utils as sw  # noqa: E402
from trinity.common.workflows.envs.lookahead_opd.scienceworld import COMPLETED_SCORE  # noqa: E402

SPLIT = "test"


def _load_tasks(path: str, start: Optional[int], end: Optional[int]) -> List[dict]:
    tasks = [json.loads(l)["task_desc"] for l in open(path) if l.strip()]
    return [{"task_desc": t, "index": i} for i, t in enumerate(tasks)][start:end]


def _is_context_overflow(e: Exception) -> bool:
    """vLLM's 400 for a prompt + max_tokens over the server's context (model_client re-raises it)."""
    return "maximum context length" in str(e)


def run_episode(env, model_client, max_steps: int, mock: bool) -> dict:
    """The training workflow's rollout, student only. A turn whose prompt (the full
    conversation) no longer fits the server's context ends the game there, keeping the best score
    so far, as a game that runs out of steps (context_overflow = True)."""
    observation, info, task_description = sw._reset_scienceworld_env(env)
    history, memory, actions, step_log = [], [], [], []
    best_score = info.get("score", 0)
    rounds, done, overflow = max_steps, False, False
    for r in range(max_steps):
        user_content, format_obs = sw.build_user_content(env, observation, history, task_description, r)
        memory = memory + [{"role": "user", "content": user_content}]
        avail = {"commands": env.get_valid_action_object_combinations()} if mock else None
        try:
            text, raw = model_client.chat(memory, available_actions=avail)
        except RuntimeError as e:
            if not _is_context_overflow(e):
                raise
            rounds, overflow = r, True
            break
        memory = memory + [{"role": "assistant", "content": text}]
        action = sw.parse_action(text)
        actions.append(action)
        history.append(sw._format_history(format_obs, r + 1, action))
        observation, reward, done, info = env.step(action)
        best_score = max(best_score, info.get("score", best_score + reward))
        finish = raw["choices"][0].get("finish_reason") if raw and raw.get("choices") else None
        step_log.append({"step": r, "response": text, "finish_reason": finish, "parsed_action": action,
                         "score": info.get("score"), "done": done})
        if done:
            rounds = r + 1
            break
    return {"task_description": task_description, "actions": actions, "env_rounds": rounds, "env_done": bool(done),
            "context_overflow": overflow, "score": best_score / 100.0, "success": best_score >= COMPLETED_SCORE,
            "step_log": step_log}


def _worker(worker_id: int, tasks: List[dict], args, shard_path: str):
    t0 = time.time()
    client = ChatModel(base_url=args.base_url, api_key=args.api_key, model=args.model,
                       temperature=args.temperature, max_tokens=args.max_tokens, top_p=args.top_p,
                       top_k=args.top_k, min_p=args.min_p, enable_thinking=args.enable_thinking,
                       mock=args.mock_model)
    with open(shard_path, "w") as f:
        for i, task in enumerate(tasks):
            spec = json.loads(task["task_desc"])
            rec = {"split": SPLIT, "index": task["index"], "task_name": spec["task_name"], "var_num": spec["var_num"]}
            env = None
            try:
                env = sw._create_scienceworld_env(task["task_desc"], max_env_steps=args.max_env_steps)
                rec.update(run_episode(env, client, args.max_env_steps, args.mock_model), error=None)
            except Exception as e:  # noqa: BLE001 - one bad game must not end the shard
                print(f"[worker {worker_id}] ERROR on {rec['task_name']} var {rec['var_num']}: {type(e).__name__}: {e}",
                      file=sys.stderr)
                rec.update(actions=[], env_rounds=0, env_done=False, score=0.0, success=False, step_log=[],
                           error=f"{type(e).__name__}: {e}")
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
            if not args.keep_step_log:
                rec.pop("step_log", None)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (i + 1) % 5 == 0 or i + 1 == len(tasks):
                print(f"[worker {worker_id}] {i + 1}/{len(tasks)} games done (last: {rec['task_name']}, "
                      f"score {rec['score']:.2f}, success {rec['success']}, error {rec['error']})")
    print(f"[worker {worker_id}] finished shard in {time.time() - t0:.1f}s")


def _write_summary(records: List[dict], summary_path: str, label: str) -> dict:
    n = len(records)
    per_task = {}
    for name in sorted({r["task_name"] for r in records}):
        rs = [r for r in records if r["task_name"] == name]
        per_task[name] = {"n": len(rs), "success_rate": sum(r["success"] for r in rs) / len(rs),
                          "mean_score": statistics.mean(r["score"] for r in rs)}
    d = {
        "n_tasks": n,
        "n_success": sum(r["success"] for r in records),
        "success_rate": sum(r["success"] for r in records) / n if n else 0.0,
        "mean_score": statistics.mean(r["score"] for r in records) if n else 0.0,
        "n_errors": sum(1 for r in records if r.get("error")),
        "n_context_overflow": sum(1 for r in records if r.get("context_overflow")),
    }
    summary = {"checkpoint": label, "per_split": {SPLIT: d}, "per_task_type": per_task}
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", required=True)
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--model", required=True)
    p.add_argument("--checkpoint-label", default=None)
    p.add_argument("--tasks-jsonl", default=os.path.join(_THIS_DIR, "../../../../data/scienceworld/test.jsonl"))
    p.add_argument("--task-start", type=int, default=None)
    p.add_argument("--task-end", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-env-steps", type=int, default=30)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--output", required=True)
    p.add_argument("--summary", required=True)
    p.add_argument("--mock-model", action="store_true", help="random valid actions, no server needed")
    p.add_argument("--no-keep-step-log", dest="keep_step_log", action="store_false")
    args = p.parse_args()

    tasks = _load_tasks(args.tasks_jsonl, args.task_start, args.task_end)
    if not tasks:
        raise SystemExit(f"no tasks in {args.tasks_jsonl}[{args.task_start}:{args.task_end}]")
    n_workers = max(1, min(args.workers, len(tasks)))
    shards = [tasks[i::n_workers] for i in range(n_workers)]
    shard_paths = [f"{args.output}.shard{i}.tmp" for i in range(n_workers)]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    print(f"Evaluating model={args.model!r} on {len(tasks)} ScienceWorld games with {n_workers} worker(s), "
          f"max_env_steps={args.max_env_steps}, temperature={args.temperature}, max_tokens={args.max_tokens}, "
          f"mock={args.mock_model}")

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_worker, args=(i, s, args, sp)) for i, (s, sp) in enumerate(zip(shards, shard_paths))]
    for proc in procs:
        proc.start()
    failed = False
    for proc in procs:
        proc.join()
        failed |= proc.exitcode != 0
    records = []
    for sp in shard_paths:
        if os.path.exists(sp):
            records += [json.loads(l) for l in open(sp) if l.strip()]
            os.remove(sp)
    records.sort(key=lambda r: r["index"])
    with open(args.output, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
    d = _write_summary(records, args.summary, args.checkpoint_label or args.model)["per_split"][SPLIT]
    errored = f", {d['n_errors']} errored" if d["n_errors"] else ""
    errored += f", {d['n_context_overflow']} ended on a full context" if d["n_context_overflow"] else ""
    print(f"Done: {len(records)}/{len(tasks)} games -> {args.output}\n"
          f"  success {d['n_success']}/{d['n_tasks']} = {d['success_rate']:.1%}, mean score {d['mean_score']:.3f}{errored}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
