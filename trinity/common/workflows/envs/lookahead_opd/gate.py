"""The LookAheadOPD gate: which student turns get the teacher's on-policy distillation signal, and how much.

For every student turn t the teacher is asked

  (A) agreement: "would you choose this exact action for the current step?" (YES_NO_ADDENDUM, on the
      conversation up to turn t's user message). If the teacher agrees, the turn gets weight 0.

and, only for the turns it disagrees with ("No" or an unparseable answer), both

  (B) student look-ahead: "do the student's next `window_size` steps make progress toward the task?"
      (PROGRESS_ADDENDUM, on the conversation through turn t + window_size - 1). The trajectory's final
      `window_size` turns have no such window; there the episode outcome decides (no success = no progress).
  (C) teacher look-ahead: the game is replayed up to turn t in a fresh copy, the teacher plays
      `window_size` steps from there itself (starting with its own action for turn t, continuing the
      student's conversation with the same prompts), and is asked the progress question about its own
      steps. If the game ends inside the window, the outcome decides and nothing is asked.

The weight of a disagreed turn, with S = "the student is NOT making progress" (B) and T = "the teacher IS
making progress" (C):

    S and T   -> both_weight          (1.0)  the student is stuck and the teacher does better from here
    T only    -> teacher_only_weight  (0.7)  both progress
    S only    -> student_only_weight  (0.3)  neither progresses
    neither   -> neither_weight       (0.0)  the student progresses and the teacher does not

Unknowns never lower the weight: an unparseable (B) counts as S, an unparseable (C) or a failed look-ahead
counts as T.

The weight w scales the turn's OPD advantage: the trainer stores `student + w * (teacher - student)` as the
turn's teacher log-probs, so the shared multi_turn_opd advantage kl_coef * (teacher - student) becomes
w times the plain OPD advantage, and w = 0 gives no update on that turn.

Where it runs: the explorer only plays the game (lookahead_opd/{alfworld,scienceworld,webshop}.py) and
attaches `payload(...)` to each turn. The trainer asks (A), (B), (C) and scores the turns for the turns it
samples into a training batch (trinity/trainer/tunix/teacher_gate.py), so most turns an explore step
produces never cost a teacher call. Every teacher look-ahead replays a game, so it runs as its own Ray task
(`lookahead_task`).
"""

import asyncio
import importlib
import json
import string
import zlib
from typing import Awaitable, Callable, Dict, List, Optional

PAYLOAD_KEY = "lookahead_opd"
DEFAULT_WINDOW_SIZE = 5
CALL_ATTEMPTS = 3

# Defaults of the paper's runs (configs/*/lookahead_opd_*.yaml)
DEFAULT_WEIGHTS = {
    "both_weight": 1.0,
    "teacher_only_weight": 0.7,
    "student_only_weight": 0.3,
    "neither_weight": 0.0,
}

# The module (lookahead_opd.<env>) that implements `teacher_lookahead` for each payload's "env"
ENV_MODULES = {
    "alfworld": "trinity.common.workflows.envs.lookahead_opd.alfworld",
    "scienceworld": "trinity.common.workflows.envs.lookahead_opd.scienceworld",
    "webshop": "trinity.common.workflows.envs.lookahead_opd.webshop",
}

YES_NO_ADDENDUM = """

Now suppose the action chosen for the current step is:
{student_action}

Would you choose this exact action for the current step? First briefly reason \
about whether it is the best available action, then give your final decision \
wrapped in <answer></answer> tags -- either <answer>Yes</answer> or \
<answer>No</answer>. Do not output any other text besides your reasoning and \
the final answer."""

PROGRESS_ADDENDUM = """Based on the {window_len} step(s) shown above (steps \
{start_step}-{end_step}), do you think the agent is making progress toward \
completing the task? First briefly reason about whether those particular \
steps help advance the goal, then give your final decision wrapped in \
<answer></answer> tags -- either <answer>Yes</answer> or <answer>No</answer>. \
Do not output any other text besides your reasoning and the final answer."""

# ask(messages, temperature, max_tokens) -> the teacher's response text
Ask = Callable[[List[Dict[str, str]], float, int], Awaitable[str]]


def parse_yes_no(response: str) -> Optional[bool]:
    """The first word inside the last <answer></answer> (or of the whole response if there is no tag):
    True for "yes", False for "no", None otherwise."""
    try:
        content = (
            response.rsplit("<answer>", 1)[-1].split("</answer>")[0]
            if "<answer>" in response
            else response
        )
        words = content.strip().lower().split()
        if not words:
            return None
        first = words[0].strip(string.punctuation)
        if first == "yes":
            return True
        if first == "no":
            return False
        return None
    except Exception:
        return None


def agree_messages(memory: List[Dict[str, str]], step: int, action: str) -> List[Dict[str, str]]:
    """(A) for turn `step` (0-indexed): the conversation up to and including that turn's user message,
    with the yes/no question appended to it (the student's own response is not shown)."""
    context = memory[: 2 * step + 1]
    return context[:-1] + [
        {
            "role": "user",
            "content": context[-1]["content"] + YES_NO_ADDENDUM.format(student_action=action),
        }
    ]


def progress_messages(
    memory: List[Dict[str, str]], window_start: int, window_end: int
) -> List[Dict[str, str]]:
    """The progress question for steps [window_start, window_end] (0-indexed, inclusive): the
    conversation through the end of the window, then the question."""
    context = memory[: 2 * (window_end + 1)]
    return context + [
        {
            "role": "user",
            "content": PROGRESS_ADDENDUM.format(
                window_len=window_end - window_start + 1,
                start_step=window_start + 1,
                end_step=window_end + 1,
            ),
        }
    ]


def lookahead_weight(
    gate: Dict, disagree: bool, student_not_progress: Optional[bool], teacher_progress: Optional[bool]
) -> float:
    """OPD weight of one turn (see the module docstring). None = unknown: counts as S for the student,
    as T for the teacher."""
    if not disagree:
        return 0.0
    s, t = student_not_progress is not False, teacher_progress is not False
    if s and t:
        return gate["both_weight"]
    if t:
        return gate["teacher_only_weight"]
    if s:
        return gate["student_only_weight"]
    return gate["neither_weight"]


def weight_class(disagree: bool, student_not_progress: Optional[bool], teacher_progress: Optional[bool]) -> str:
    """"agree", or which of the four cases of a disagreed turn applies: "both", "teacher_only",
    "student_only", "neither"."""
    if not disagree:
        return "agree"
    s, t = student_not_progress is not False, teacher_progress is not False
    return "both" if s and t else "teacher_only" if t else "student_only" if s else "neither"


def payload(
    env: str,
    memory: List[Dict[str, str]],
    step: int,
    actions: List[str],
    outcome: float,
    gate: Dict,
    task: str,
) -> Dict:
    """What the trainer needs to gate and score turn `step` of an episode by itself.

    `outcome` is 1.0 if the episode counts as progress for the student's final window (ALFWorld: won,
    ScienceWorld: score 100, WebShop: purchase score >= progress_score), else 0.0. The conversation is kept
    through the end of the turn's student window, or only up to its user message for the final
    `window_size` turns (which use the outcome); zlib-compressed, since the full conversation repeats in
    every turn of an episode. `task` and the student's earlier actions let the trainer replay the game for
    the teacher's look-ahead.
    """
    n_steps = len(actions)
    has_window = step + gate["window_size"] < n_steps
    keep = 2 * (step + gate["window_size"]) if has_window else 2 * step + 1
    return {
        "env": env,
        "step": step,
        "n_steps": n_steps,
        "action": actions[step],
        "outcome": float(outcome),
        "has_window": has_window,
        "memory_z": zlib.compress(json.dumps(memory[:keep]).encode()),
        "gate": gate,
        "task": task,
        "prior_actions": list(actions[:step]),
    }


def payload_memory(p: Dict) -> List[Dict[str, str]]:
    return json.loads(zlib.decompress(p["memory_z"]))


def env_module(env: str):
    return importlib.import_module(ENV_MODULES[env])


def lookahead_task(teacher_name: str, namespace: str, p: Dict) -> Dict:
    """(C) for one turn, as a Ray task: reaches the teacher engine by its actor name, so the game replay
    runs in this task's own process."""
    import ray

    teacher = ray.get_actor(teacher_name, namespace=namespace)

    async def ask(messages, temperature, max_tokens):
        for attempt in range(CALL_ATTEMPTS):
            try:
                responses = await teacher.chat.remote(
                    messages, lora_request=None, temperature=temperature, max_tokens=max_tokens, n=1
                )
                return responses[0].response_text or ""
            except Exception:
                if attempt == CALL_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(2.0)

    context = payload_memory(p)[: 2 * p["step"] + 1]
    return asyncio.run(
        env_module(p["env"]).teacher_lookahead(
            ask, p["task"], context, p["prior_actions"], p["step"], p["gate"], p["action"]
        )
    )


def lookahead_metrics(results: List[Optional[Dict]], prefix: str) -> Dict[str, float]:
    """Summary of a batch's teacher look-aheads (None = failed)."""
    ok = [r for r in results if r is not None]
    metrics = {prefix + "count": len(results), prefix + "errors": len(results) - len(ok)}
    if ok:
        n = len(ok)
        asked = [r for r in ok if r["asked"]]
        metrics.update({
            prefix + "no_progress_rate": sum(r["progress"] is False for r in ok) / n,
            prefix + "finished_rate": sum(r["finished"] for r in ok) / n,
            prefix + "steps_mean": sum(r["steps"] for r in ok) / n,
            prefix + "same_first_action_rate": sum(r["same_first_action"] for r in ok) / n,
            prefix + "replay_ok_rate": sum(r["replay_ok"] for r in ok) / n,
        })
        if asked:
            metrics[prefix + "progress_parse_success_rate"] = (
                1.0 - sum(r["progress"] is None for r in asked) / len(asked)
            )
    return metrics
