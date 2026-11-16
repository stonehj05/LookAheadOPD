"""LookAheadOPD on WebShop (see gate.py for the method).

The rollout is TCOD's WebShop setup (session id = task, the admissible actions listed every turn, an invalid
action answered with validate_action's message without stepping the game, reward = the shop's score of the
purchase, 0 without one) with DASH-OPD's WebShop prompts (webshop_utils.build_webshop_prompt: history
length 2, no history for the first 3 turns, the full conversation as memory).

Games are played on shared WebShop servers (webshop_env_server.py): one catalog per server actor instead of
one per env. The teacher's look-ahead replays the student's earlier actions on a new game of the same
session (WebShop is deterministic given the session and the servers' seeded prices).

A WebShop game ends when the agent buys something, scored in [0, 1] by how well the product matches the
goal. In the gate a purchase scoring at least `progress_score` (workflow arg, default 1.0; the paper's runs
use 0.5) counts as progress: for the student's final window, and for a teacher look-ahead that buys
something (then nothing is asked). The look-ahead window is not cut by max_env_steps.

Episode metrics (logged as batch means): env_score = the purchase's score (also the reward, 0 without a
purchase), env_success = 1 if it is 1.0.
"""

import time
from typing import Dict, List

from trinity.common.experience import Experience
from trinity.common.workflows import WORKFLOWS
from trinity.common.workflows.envs.lookahead_opd.gate import Ask, parse_yes_no, progress_messages
from trinity.common.workflows.envs.lookahead_opd.webshop_env_server import RemoteWebShopEnv
from trinity.common.workflows.envs.lookahead_opd.webshop_utils import (
    HISTORY_LENGTH,
    INITIAL_NO_HISTORY_TURNS,
    _extract_task_description,
    _format_available_actions,
    _format_history,
    build_webshop_prompt,
    format_observation,
    parse_action,
    validate_action,
)
from trinity.common.workflows.envs.lookahead_opd.workflow import LookAheadOPDWorkflow

ENV = "webshop"
COMPLETED_SCORE = 1.0


def build_user_content(observation: str, available_actions: Dict, history: List[str], task_description: str, r: int):
    """The user message of step `r` (0-indexed) and the formatted observation."""
    format_obs = format_observation(observation)
    user_content = build_webshop_prompt(
        task_description=task_description,
        step_count=r,
        history=history,
        history_length=HISTORY_LENGTH,
        initial_no_history_turns=INITIAL_NO_HISTORY_TURNS,
        current_observation=format_obs,
        available_actions=_format_available_actions(available_actions),
        use_initial_no_history_template=True,
    )
    return user_content, format_obs


async def play(env: RemoteWebShopEnv, observation: str, action: str):
    """One step: an invalid action does not reach the game; its observation is validate_action's message.
    Returns (observation, reward, done)."""
    valid, error_message = validate_action(action, env.get_available_actions())
    if not valid:
        return error_message, 0.0, False
    observation, reward, done, _ = await env.step(action)
    return observation, reward, done


async def replay_game(session: int, actions: List[str], step: int):
    """A new game of the session after the first `step` of `actions`.
    Returns (env, observation, history, task_description, n_replayed, done)."""
    env = RemoteWebShopEnv(f"lookahead/{session}/{step}")
    observation = await env.reset(session=session)
    task_description = _extract_task_description(observation)
    history: List[str] = []
    done = False
    for k, action in enumerate(actions[:step]):
        history.append(_format_history(format_observation(observation), k + 1, action))
        observation, _, done = await play(env, observation, action)
        if done:
            break
    return env, observation, history, task_description, len(history), done


async def teacher_lookahead(
    ask: Ask,
    task_desc: str,
    context: List[Dict[str, str]],
    prior_actions: List[str],
    step: int,
    gate: Dict,
    student_action: str = "",
) -> Dict:
    """(C): the teacher plays `gate["window_size"]` steps from step `step` (0-indexed) of a replayed copy
    of the game and judges them (as alfworld.teacher_lookahead)."""
    env, observation, history, task_description, n_replayed, done = await replay_game(
        int(task_desc), prior_actions, step
    )
    try:
        progress_score = gate.get("progress_score", COMPLETED_SCORE)
        if done or n_replayed != step:
            raise RuntimeError(f"replay ended after {n_replayed} of {step} steps (done={done})")
        user_content, format_obs = build_user_content(
            observation, env.get_available_actions(), history, task_description, step
        )
        replay_ok = user_content == context[-1]["content"]
        ctx = list(context)
        teacher_actions: List[str] = []
        ended, completed, progressed = False, False, False
        for k in range(gate["window_size"]):
            r = step + k
            if k > 0:
                user_content, format_obs = build_user_content(
                    observation, env.get_available_actions(), history, task_description, r
                )
                ctx.append({"role": "user", "content": user_content})
            text = await ask(ctx, gate["teacher_rollout_temperature"], gate["teacher_rollout_max_tokens"])
            ctx.append({"role": "assistant", "content": text})
            action = parse_action(text)
            teacher_actions.append(action)
            history.append(_format_history(format_obs, r + 1, action))
            observation, reward, done = await play(env, observation, action)
            if done:
                ended, completed, progressed = True, reward >= COMPLETED_SCORE, reward >= progress_score
                break
    finally:
        await env.close()

    if ended:  # the game is over: progress by outcome (a purchase scoring >= progress_score), no question
        progress, progress_text = progressed, ""
    else:
        progress_text = await ask(
            progress_messages(ctx, step, step + len(teacher_actions) - 1),
            gate["progress_temperature"],
            gate["progress_max_tokens"],
        )
        progress = parse_yes_no(progress_text)
    return {
        "progress": progress,
        "asked": not ended,
        "finished": completed,
        "steps": len(teacher_actions),
        "actions": teacher_actions,
        "same_first_action": teacher_actions[0] == student_action,
        "replay_ok": replay_ok,
        "progress_text_len": len(progress_text),
    }


@WORKFLOWS.register_module("LookAheadOPD_webshop_workflow")
class LookAheadOPDWebshopWorkflow(LookAheadOPDWorkflow):
    ENV = ENV
    DEFAULT_MAX_ENV_STEPS = 15

    def __init__(self, *, task, model, auxiliary_models=None):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.progress_score = float(task.workflow_args.get("progress_score", COMPLETED_SCORE))
        self.gate["progress_score"] = self.progress_score
        self._env = None

    def _outcome(self) -> float:
        return 1.0 if self._final_reward >= self.progress_score else 0.0

    def _completed(self) -> float:
        return 1.0 if self._final_reward >= COMPLETED_SCORE else 0.0

    def _episode_metrics(self) -> Dict[str, float]:
        return {"env_score": self._final_reward, "env_success": self._completed()}

    async def run_async(self) -> List[Experience]:
        # one game on a shared server per workflow instance, reused across tasks
        if self._env is None:
            self._env = RemoteWebShopEnv(f"runner/{id(self)}/{time.time()}")
        return await self._run_episode(self._env)

    async def _rollout(self, env):
        session_id = int(self.task_desc)
        observation = await env.reset(session=session_id)
        task_description = _extract_task_description(observation)
        self._env_done = False
        self._env_rounds = 0
        self._final_reward = 0.0
        history: List[str] = []
        turn_responses: List[Experience] = []
        actions: List[str] = []
        memory: List[Dict[str, str]] = []
        kwargs = self._student_kwargs()

        for r in range(self.max_env_steps):
            user_content, format_obs = build_user_content(
                observation, env.get_available_actions(), history, task_description, r
            )
            memory = memory + [{"role": "user", "content": user_content}]
            response = await self._student_turn(memory, kwargs)
            response_text = response.response_text or ""
            memory = memory + [{"role": "assistant", "content": response_text}]

            action = parse_action(response_text)
            turn_responses.append(response)
            actions.append(action)

            history.append(_format_history(format_obs, r + 1, action))
            observation, reward, done = await play(env, observation, action)
            if done:
                self._env_done = True
                self._env_rounds = r + 1
                self._final_reward = float(reward)
                break
        else:
            self._env_rounds = self.max_env_steps
        return turn_responses, actions, memory
