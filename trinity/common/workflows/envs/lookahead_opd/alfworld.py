"""LookAheadOPD on ALFWorld (see gate.py for the method).

The rollout is TCOD's ALFWorld setup with the full conversation as memory: one user message per turn
(observation, admissible actions, and from the third turn on the last HISTORY_LENGTH steps), reward 1 if
the game is won within max_env_steps, else 0. A won game counts as progress for the student's final
window; a teacher look-ahead that wins the game is progress without asking.
"""

from typing import Dict, List

from trinity.common.experience import Experience
from trinity.common.workflows import WORKFLOWS
from trinity.common.workflows.envs.lookahead_opd.alfworld_utils import (
    ALFWORLD_TEMPLATE,
    ALFWORLD_TEMPLATE_NO_HIS,
    HISTORY_LENGTH,
    _create_alfworld_env,
    _extract_task,
    _format_history,
    format_observation,
    parse_action,
    replay_game,
)
from trinity.common.workflows.envs.lookahead_opd.gate import Ask, parse_yes_no, progress_messages
from trinity.common.workflows.envs.lookahead_opd.workflow import LookAheadOPDWorkflow

ENV = "alfworld"


def build_user_content(observation: str, info: Dict, history: List[str], task_description: str, r: int):
    """The user message of step `r` (0-indexed) and the formatted observation."""
    format_obs = format_observation(observation)
    admissible_commands = info.get("admissible_commands", [])
    if admissible_commands and isinstance(admissible_commands[0], list):
        admissible_commands = admissible_commands[0]
    reformatted_admissible = "\n ".join(f"'{s}'" for s in admissible_commands if s != "help")
    if len(history) < HISTORY_LENGTH:
        user_content = ALFWORLD_TEMPLATE_NO_HIS.format(
            current_observation=format_obs,
            admissible_actions=reformatted_admissible,
        )
    else:
        user_content = ALFWORLD_TEMPLATE.format(
            task_description=task_description,
            step_count=r,
            history_length=min(HISTORY_LENGTH, len(history)),
            action_history="\n".join(history[-HISTORY_LENGTH:]),
            current_step=r + 1,
            current_observation=format_obs,
            admissible_actions=reformatted_admissible,
        )
    return user_content, format_obs


async def teacher_lookahead(
    ask: Ask,
    game_file: str,
    context: List[Dict[str, str]],
    prior_actions: List[str],
    step: int,
    gate: Dict,
    student_action: str = "",
) -> Dict:
    """(C): the teacher plays `gate["window_size"]` steps from step `step` (0-indexed) of a replayed copy
    of the game and judges them. `context` is the student's conversation through step `step`'s user
    message; `prior_actions` are the student's actions of steps 0..step-1. The window is not cut by
    max_env_steps. Returns `progress` (True / False / None = unparseable) and what happened on the way."""
    env, observation, info, history, task_description, n_replayed, done = replay_game(
        game_file, list(prior_actions[:step]), step
    )
    try:
        if done or n_replayed != step:
            raise RuntimeError(f"replay ended after {n_replayed} of {step} steps (done={done})")
        user_content, format_obs = build_user_content(observation, info, history, task_description, step)
        replay_ok = user_content == context[-1]["content"]
        # The teacher continues the conversation the student actually had.
        ctx = list(context)
        teacher_actions: List[str] = []
        finished = False
        for k in range(gate["window_size"]):
            r = step + k
            if k > 0:
                user_content, format_obs = build_user_content(observation, info, history, task_description, r)
                ctx.append({"role": "user", "content": user_content})
            text = await ask(ctx, gate["teacher_rollout_temperature"], gate["teacher_rollout_max_tokens"])
            ctx.append({"role": "assistant", "content": text})
            action = parse_action(text)
            teacher_actions.append(action)
            history.append(_format_history(format_obs, r + 1, action))
            observation, _, done, info = env.step(action)
            if done:
                finished = True
                break
    finally:
        env.close()

    if finished:  # the teacher completed the task: progress by outcome, no question needed
        progress, progress_text = True, ""
    else:
        progress_text = await ask(
            progress_messages(ctx, step, step + len(teacher_actions) - 1),
            gate["progress_temperature"],
            gate["progress_max_tokens"],
        )
        progress = parse_yes_no(progress_text)
    return {
        "progress": progress,
        "asked": not finished,
        "finished": finished,
        "steps": len(teacher_actions),
        "actions": teacher_actions,
        "same_first_action": teacher_actions[0] == student_action,
        "replay_ok": replay_ok,
        "progress_text_len": len(progress_text),
    }


@WORKFLOWS.register_module("LookAheadOPD_alfworld_workflow")
class LookAheadOPDAlfworldWorkflow(LookAheadOPDWorkflow):
    ENV = ENV

    def _outcome(self) -> float:
        return self._final_reward  # 1.0 = won

    async def run_async(self) -> List[Experience]:
        env = _create_alfworld_env(self.task_desc)
        try:
            return await self._run_episode(env)
        finally:
            env.close()

    async def _rollout(self, env):
        observation, info = env.reset()
        self._env_done = False
        self._env_rounds = 0
        task_description = _extract_task(observation)
        history: List[str] = []
        turn_responses: List[Experience] = []
        actions: List[str] = []
        memory: List[Dict[str, str]] = []
        kwargs = self._student_kwargs()

        for r in range(self.max_env_steps):
            user_content, format_obs = build_user_content(observation, info, history, task_description, r)
            memory = memory + [{"role": "user", "content": user_content}]
            response = await self._student_turn(memory, kwargs)
            response_text = response.response_text or ""
            memory = memory + [{"role": "assistant", "content": response_text}]

            action = parse_action(response_text)
            turn_responses.append(response)
            actions.append(action)

            history.append(_format_history(format_obs, r + 1, action))
            observation, reward, done, info = env.step(action)
            if done:
                self._env_done = True
                self._env_rounds = r + 1
                self._final_reward = 1.0
                break
        else:
            self._env_rounds = self.max_env_steps
            self._final_reward = 0.0  # failure: exhausted max steps
        return turn_responses, actions, memory
