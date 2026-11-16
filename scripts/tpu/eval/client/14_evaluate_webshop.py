#!/usr/bin/env python
"""Evaluate a checkpoint on WebShop's test sessions (data/webshop/test.jsonl: session ids 4096-4195, the same
as TCOD's and DASH-OPD's), served by vLLM, with the rollout LookAheadOPD trains on: DASH-OPD's WebShop
prompts (build_webshop_prompt: no history summary in turns 1-3, the last 2 observation-action pairs from
turn 4) and the full conversation as memory, built by the same functions as the training workflow
(lookahead_opd/webshop.py: build_user_content; an invalid action does not reach the game).

Per game: score = the purchase's score (0 without a purchase within the step limit), success = score 1,
as DASH-OPD's WebShop benchmark (eval_workflow.py). A game whose conversation outgrows the server's
context ends at that turn with score 0 (context_overflow). Eval settings default to DASH-OPD's WebShop
benchmark (opd_bench.yaml): temperature 0.4, max_tokens 4096, 15 game steps, thinking off.

The shop is one in-process WebShopEnvServer (webshop_env_server.py, the same seed as training's servers: the
same goals and prices); its games run concurrently (--workers), the model calls in threads and every
WebShop call in the main thread (its search engine runs in a JVM).

  python 14_evaluate_webshop.py --base-url http://localhost:8081/v1 --model step150 \\
      --tasks-jsonl ~/LookAheadOPD/data/webshop/test.jsonl --output out.jsonl --summary out.summary.json

Run one job per chip and avg@4 with ../evaluate_webshop_avg4.sh.
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from typing import List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(_THIS_DIR, "../../../..")))  # the repo

from model_client import ChatModel  # noqa: E402
from trinity.common.workflows.envs.lookahead_opd.webshop import COMPLETED_SCORE, build_user_content  # noqa: E402
from trinity.common.workflows.envs.lookahead_opd.webshop_env_server import DEFAULT_SEED, WebShopEnvServer  # noqa: E402
from trinity.common.workflows.envs.lookahead_opd.webshop_utils import (  # noqa: E402
    _extract_task_description,
    _format_history,
    parse_action,
    validate_action,
)

SPLIT = "test"


def _load_tasks(path: str, start: Optional[int], end: Optional[int]) -> List[dict]:
    sessions = [int(json.loads(l)["task_id"]) for l in open(path) if l.strip()]
    return [{"session": s, "index": i} for i, s in enumerate(sessions)][start:end]


def _is_context_overflow(e: Exception) -> bool:
    """vLLM's 400 for a prompt + max_tokens over the server's context (model_client re-raises it)."""
    return "maximum context length" in str(e)


async def run_episode(server: WebShopEnvServer, client: ChatModel, session: int, max_steps: int) -> dict:
    """The training workflow's rollout, student only."""
    env_id = server.new_env()
    try:
        observation, available = server.reset(env_id, session)
        task_description = _extract_task_description(observation)
        history, memory, actions, step_log = [], [], [], []
        rounds, done, overflow, score = max_steps, False, False, 0.0
        for r in range(max_steps):
            user_content, format_obs = build_user_content(observation, available, history, task_description, r)
            memory = memory + [{"role": "user", "content": user_content}]
            try:
                text, raw = await asyncio.to_thread(client.chat, memory)
            except RuntimeError as e:
                if not _is_context_overflow(e):
                    raise
                rounds, overflow = r, True
                break
            memory = memory + [{"role": "assistant", "content": text}]
            action = parse_action(text)
            actions.append(action)
            history.append(_format_history(format_obs, r + 1, action))
            # play() of the workflow: an invalid action does not reach the game
            valid, error_message = validate_action(action, available)
            if valid:
                observation, reward, done, available = server.step(env_id, action)
            else:
                observation, reward, done = error_message, 0.0, False
            finish = raw["choices"][0].get("finish_reason") if raw and raw.get("choices") else None
            step_log.append({"step": r, "response": text, "finish_reason": finish, "parsed_action": action,
                             "valid": valid, "reward": reward, "done": done})
            if done:
                rounds, score = r + 1, float(reward)
                break
        return {"task_description": task_description, "actions": actions, "env_rounds": rounds, "env_done": bool(done),
                "context_overflow": overflow, "score": score, "success": score >= COMPLETED_SCORE,
                "step_log": step_log}
    finally:
        server.close(env_id)


async def _run_all(tasks: List[dict], args, out_path: str) -> List[dict]:
    t0 = time.time()
    server = WebShopEnvServer(seed=args.seed)
    print(f"shop loaded ({len(server.server.goals)} goals) in {time.time() - t0:.0f}s", flush=True)
    client = ChatModel(base_url=args.base_url, api_key=args.api_key, model=args.model,
                       temperature=args.temperature, max_tokens=args.max_tokens, top_p=args.top_p,
                       top_k=args.top_k, min_p=args.min_p, enable_thinking=args.enable_thinking)
    limit = asyncio.Semaphore(max(1, args.workers))
    records: List[dict] = []
    f = open(out_path, "w")

    async def one(task):
        async with limit:
            rec = {"split": SPLIT, "index": task["index"], "session": task["session"]}
            try:
                rec.update(await run_episode(server, client, task["session"], args.max_env_steps), error=None)
            except Exception as e:  # noqa: BLE001 - one bad game must not end the job
                print(f"ERROR on session {task['session']}: {type(e).__name__}: {e}", file=sys.stderr)
                rec.update(actions=[], env_rounds=0, env_done=False, context_overflow=False, score=0.0,
                           success=False, step_log=[], error=f"{type(e).__name__}: {e}")
            if not args.keep_step_log:
                rec.pop("step_log", None)
            records.append(rec)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if len(records) % 5 == 0 or len(records) == len(tasks):
                print(f"{len(records)}/{len(tasks)} games done (last: session {rec['session']}, "
                      f"score {rec['score']:.2f}, success {rec['success']}, error {rec['error']})", flush=True)

    await asyncio.gather(*[one(t) for t in tasks])
    f.close()
    print(f"finished in {time.time() - t0:.1f}s")
    return records


def _write_summary(records: List[dict], summary_path: str, label: str) -> dict:
    n = len(records)
    d = {
        "n_tasks": n,
        "n_success": sum(r["success"] for r in records),
        "success_rate": sum(r["success"] for r in records) / n if n else 0.0,
        "mean_score": statistics.mean(r["score"] for r in records) if n else 0.0,
        "n_errors": sum(1 for r in records if r.get("error")),
        "n_context_overflow": sum(1 for r in records if r.get("context_overflow")),
        "mean_env_rounds": statistics.mean(r["env_rounds"] for r in records) if n else 0.0,
    }
    summary = {"checkpoint": label, "per_split": {SPLIT: d}}
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", required=True)
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--model", required=True)
    p.add_argument("--checkpoint-label", default=None)
    p.add_argument("--tasks-jsonl", default=os.path.join(_THIS_DIR, "../../../../data/webshop/test.jsonl"))
    p.add_argument("--task-start", type=int, default=None)
    p.add_argument("--task-end", type=int, default=None)
    p.add_argument("--workers", type=int, default=8, help="games played at once")
    p.add_argument("--max-env-steps", type=int, default=15)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="the shop's seed (training's servers: 0)")
    p.add_argument("--output", required=True)
    p.add_argument("--summary", required=True)
    p.add_argument("--no-keep-step-log", dest="keep_step_log", action="store_false")
    args = p.parse_args()

    tasks = _load_tasks(args.tasks_jsonl, args.task_start, args.task_end)
    if not tasks:
        raise SystemExit(f"no tasks in {args.tasks_jsonl}[{args.task_start}:{args.task_end}]")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    print(f"Evaluating model={args.model!r} on {len(tasks)} WebShop sessions, {args.workers} at once, "
          f"max_env_steps={args.max_env_steps}, temperature={args.temperature}, max_tokens={args.max_tokens}")
    records = asyncio.run(_run_all(tasks, args, args.output + ".tmp"))
    records.sort(key=lambda r: r["index"])
    with open(args.output, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
    os.remove(args.output + ".tmp")
    d = _write_summary(records, args.summary, args.checkpoint_label or args.model)["per_split"][SPLIT]
    extra = f", {d['n_errors']} errored" if d["n_errors"] else ""
    extra += f", {d['n_context_overflow']} ended on a full context" if d["n_context_overflow"] else ""
    print(f"Done: {len(records)}/{len(tasks)} games -> {args.output}\n"
          f"  success {d['n_success']}/{d['n_tasks']} = {d['success_rate']:.1%}, mean score {d['mean_score']:.3f}{extra}")


if __name__ == "__main__":
    main()
