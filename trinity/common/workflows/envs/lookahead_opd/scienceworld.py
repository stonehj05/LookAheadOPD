"""LookAheadOPD on ScienceWorld (see gate.py for the method).

The rollout is TCOD's ScienceWorld setup with DASH-OPD's prompts (scienceworld_utils.build_user_content):
the compact action list (action templates + objects), the full conversation as memory, the last 2 steps
summarized from turn 4, and reward = best score / 100.

ScienceWorld scores partial credit and also ends a game that has failed, so "the task is completed" means a
score of 100: the student's final window counts as progress only if it completed the task; a teacher
look-ahead that completes it is progress, one that ends the game otherwise is no progress (neither asks).
The teacher's look-ahead replays the game with scienceworld_utils.replay_game, which creates and plays the
copy exactly as the student's game (`teacher_lookahead_replay_ok` reports whether the replayed prompt
equals the student's).

Episode metrics (logged as batch means): env_score = best score / 100 (also the reward), env_success = 1 if
the task was completed. env_done is ScienceWorld's own end of game, which also happens on failure.
"""

from typing import Dict, List

from trinity.common.experience import Experience
from trinity.common.workflows import WORKFLOWS
from trinity.common.workflows.envs.lookahead_opd.gate import Ask, parse_yes_no, progress_messages
from trinity.common.workflows.envs.lookahead_opd.scienceworld_utils import (
    _create_scienceworld_env,
    _format_history,
    _reset_scienceworld_env,
    build_user_content,
    parse_action,
    replay_game,
)
from trinity.common.workflows.envs.lookahead_opd.workflow import LookAheadOPDWorkflow

ENV = "scienceworld"
COMPLETED_SCORE = 100


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
    env, observation, info, history, task_description, n_replayed, done = replay_game(task_desc, prior_actions, step)
    try:
        if done or n_replayed != step:
            raise RuntimeError(f"replay ended after {n_replayed} of {step} steps (done={done})")
        user_content, format_obs = build_user_content(env, observation, history, task_description, step)
        replay_ok = user_content == context[-1]["content"]
        ctx = list(context)
        teacher_actions: List[str] = []
        ended, completed = False, False
        for k in range(gate["window_size"]):
            r = step + k
            if k > 0:
                user_content, format_obs = build_user_content(env, observation, history, task_description, r)
                ctx.append({"role": "user", "content": user_content})
            text = await ask(ctx, gate["teacher_rollout_temperature"], gate["teacher_rollout_max_tokens"])
            ctx.append({"role": "assistant", "content": text})
            action = parse_action(text)
            teacher_actions.append(action)
            history.append(_format_history(format_obs, r + 1, action))
            observation, _, done, info = env.step(action)
            if done:
                ended, completed = True, info.get("score", 0) >= COMPLETED_SCORE
                break
    finally:
        env.close()

    if ended:  # the game is over: progress by outcome, no question needed
        progress, progress_text = completed, ""
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


@WORKFLOWS.register_module("LookAheadOPD_scienceworld_workflow")
class LookAheadOPDScienceworldWorkflow(LookAheadOPDWorkflow):
    ENV = ENV

    def _completed(self) -> float:
        return 1.0 if self._best_score >= COMPLETED_SCORE else 0.0

    def _outcome(self) -> float:
        return self._completed()

    def _episode_metrics(self) -> Dict[str, float]:
        return {"env_score": self._final_reward, "env_success": self._completed()}

    async def run_async(self) -> List[Experience]:
        env = _create_scienceworld_env(self.task_desc, max_env_steps=self.max_env_steps, generate_gold_path=False)
        try:
            return await self._run_episode(env)
        finally:
            env.close()

    async def _rollout(self, env):
        observation, info, task_description = _reset_scienceworld_env(env)
        self._env_done = False
        self._env_rounds = 0
        history: List[str] = []
        turn_responses: List[Experience] = []
        actions: List[str] = []
        memory: List[Dict[str, str]] = []
        best_score = info.get("score", 0)
        kwargs = self._student_kwargs()

        for r in range(self.max_env_steps):
            user_content, format_obs = build_user_content(env, observation, history, task_description, r)
            memory = memory + [{"role": "user", "content": user_content}]
            response = await self._student_turn(memory, kwargs)
            response_text = response.response_text or ""
            memory = memory + [{"role": "assistant", "content": response_text}]

            action = parse_action(response_text)
            turn_responses.append(response)
            actions.append(action)

            history.append(_format_history(format_obs, r + 1, action))
            observation, reward, done, info = env.step(action)
            best_score = max(best_score, info.get("score", best_score + reward))
            if done:
                self._env_done = True
                self._env_rounds = r + 1
                break
        else:
            self._env_rounds = self.max_env_steps

        self._best_score = best_score
        self._final_reward = best_score / 100.0
        return turn_responses, actions, memory
