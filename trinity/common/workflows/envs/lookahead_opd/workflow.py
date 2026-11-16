"""Explorer side of LookAheadOPD: the student plays an episode and every turn gets the gate's payload.

The explorer makes no teacher call. Each turn's experience carries `info[PAYLOAD_KEY]` (gate.payload),
and the trainer gates and scores the turns it samples (trinity/trainer/tunix/teacher_gate.py). The
environments subclass LookAheadOPDWorkflow with their game (`run_async`, `_rollout`) and their episode
outcome (`_outcome`).
"""

from dataclasses import asdict
from typing import Dict, List, Optional

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows import Task, Workflow
from trinity.common.workflows.envs.lookahead_opd import gate as G


class LookAheadOPDWorkflow(Workflow):
    """Rollout + payloads; see gate.py for the method."""

    ENV = ""  # gate.ENV_MODULES key
    DEFAULT_MAX_ENV_STEPS = 30

    is_async: bool = True
    can_reset: bool = True
    can_repeat: bool = False

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self.reset(task)
        args = task.workflow_args
        self.max_env_steps = args.get("max_env_steps", self.DEFAULT_MAX_ENV_STEPS)
        self.gate = {
            "window_size": args.get("progress_window_size", G.DEFAULT_WINDOW_SIZE),
            **{k: args.get(k, v) for k, v in G.DEFAULT_WEIGHTS.items()},
            # (A) agreement question
            "consistency_temperature": args.get("consistency_temperature", 0.0),
            "consistency_max_tokens": args.get("consistency_max_tokens", 512),
            # (B) and (C): the progress question
            "progress_temperature": args.get("progress_temperature", 0.0),
            "progress_max_tokens": args.get("progress_max_tokens", 512),
            # (C): the teacher's own steps
            "teacher_rollout_temperature": args.get("teacher_rollout_temperature", 0.0),
            "teacher_rollout_max_tokens": args.get("teacher_rollout_max_tokens", 512),
            # temperature of the teacher's log-prob scoring
            "temperature": args.get("temperature", 1.0),
        }

    def reset(self, task: Task):
        self.task = task
        self.format_args = task.format_args
        self.raw_task = task.raw_task
        self.task_desc = task.task_desc or "0"
        self.is_eval = task.is_eval

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.task.rollout_args.n = repeat_times
        self.run_id_base = run_id_base

    @property
    def rollout_args(self):
        return asdict(self.task.rollout_args)

    def _student_kwargs(self) -> Dict:
        kwargs = {**self.rollout_args, "n": 1}
        if kwargs.get("logprobs") is None:
            kwargs["logprobs"] = 0
        return kwargs

    async def _student_turn(self, memory: List[Dict[str, str]], kwargs: Dict) -> Experience:
        response = (await self.model.chat_async(memory, **kwargs))[0]
        if response.logprobs is None:
            raise RuntimeError(
                f"{type(self).__name__} requires the student model to return logprobs. "
                "Set rollout_args.logprobs (e.g. 0) in the task config."
            )
        return response

    def _outcome(self) -> float:
        """1.0 if the episode counts as progress for the student's final window, else 0.0."""
        raise NotImplementedError

    def _episode_metrics(self) -> Dict[str, float]:
        return {}

    async def _rollout(self, env):
        """The student's episode: returns (turn_responses, actions, memory) and sets _env_done,
        _env_rounds and _final_reward (the reward of every turn)."""
        raise NotImplementedError

    async def _run_episode(self, env) -> List[Experience]:
        turn_responses, actions, memory = await self._rollout(env)
        outcome = self._outcome()
        for i, response in enumerate(turn_responses):
            if response.info is None:
                response.info = {}
            response.info[G.PAYLOAD_KEY] = G.payload(
                self.ENV, memory, i, actions, outcome, self.gate, str(self.task_desc)
            )
            if response.metrics is None:
                response.metrics = {}
            response.reward = self._final_reward
            response.eid.run = getattr(self, "run_id_base", 0)
            response.eid.step = i
        if turn_responses:
            last = turn_responses[-1].metrics
            last["env_rounds"] = self._env_rounds
            last["env_done"] = 1.0 if self._env_done else 0.0
            last.update(self._episode_metrics())
        return turn_responses
