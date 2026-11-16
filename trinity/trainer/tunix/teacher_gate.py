"""The LookAheadOPD gate inside the trainer (see trinity/common/workflows/envs/lookahead_opd/gate.py).

The explorer only plays games; every turn carries the gate's inputs in `info[PAYLOAD_KEY]`. An explore step
produces several hundred turns and the trainer samples `train_batch_size` of them, so the teacher is asked
only for the sampled turns. `LookAheadGate.annotate` does, for a sampled batch:

  1. (A) the agreement question for every turn;
  2. for the turns the teacher disagreed with, at the same time:
       (B) the progress question on the student's next `window_size` steps (the final `window_size` turns of
           an episode use its outcome instead), and
       (C) the teacher's look-ahead: one Ray task per turn (one CPU, any node) that replays the game and lets
           the teacher play ahead (`gate.lookahead_task`);
  3. the weight of every turn (`gate.lookahead_weight`);
  4. the teacher's log-probs of the student's tokens, blended toward the student's own by the weight and
     stored as `teacher_logprobs`.

The teacher engines are the explorer's auxiliary models (created and placed by the explorer); they are reached
by their Ray actor names. Settings, all in the taskset's workflow_args:
  teacher_parallel_prompts   requests in flight per teacher engine (default 32)
  skip_zero_weight_scoring   do not score turns whose weight is 0 (default false). Their blended teacher
                             log-probs equal the student's own whatever the teacher says, so training is
                             unchanged; only the raw-KL diagnostic then covers fewer turns.
  lookahead_in_process       run the teacher look-aheads inside the trainer process, one game replay after
                             another, instead of as Ray tasks (default false; for tests and debugging)
"""

import asyncio
import time
from collections import Counter
from typing import Dict, List, Optional

import ray
import torch

from trinity.common.config import Config
from trinity.common.experience import Experience
from trinity.common.workflows.envs.lookahead_opd import gate as G
from trinity.utils.log import get_logger

DEFAULT_PARALLEL_PER_TEACHER = 32
WEIGHT_CLASSES = ("both", "teacher_only", "student_only", "neither")


class LookAheadGate:
    def __init__(self, config: Config, teachers: Optional[List] = None):
        self.logger = get_logger(__name__, in_ray_actor=True)
        # Config validation moves `explorer_input.taskset` into `tasksets`.
        explorer_input = config.buffer.explorer_input
        tasksets = ([explorer_input.taskset] if explorer_input.taskset else []) + list(explorer_input.tasksets or [])
        args = (tasksets[0].workflow_args or {}) if tasksets else {}
        self.parallel = max(1, int(args.get("teacher_parallel_prompts", DEFAULT_PARALLEL_PER_TEACHER)))
        self.skip_zero_weight_scoring = bool(args.get("skip_zero_weight_scoring", False))
        self.lookahead_in_process = bool(args.get("lookahead_in_process", False))
        # Look-ahead tasks replay games: give them the workflow runners' environment (TMPDIR on tmpfs, see
        # the configs).
        self._task_env = dict(getattr(config.explorer, "env_vars", None) or {})
        self._lookahead_remote = None
        aux = config.explorer.auxiliary_models
        self.teacher_names = [
            f"{config.explorer.name}_auxiliary_model_0_{j}" for j in range(aux[0].engine_num)
        ] if aux else []
        self.namespace = config.ray_namespace
        self.teachers = teachers  # Ray actor handles; looked up on first use
        self.student_name = f"{config.explorer.name}_rollout_model_0"
        self._marker: Optional[List[int]] = None  # see _teacher_marker
        self._prompt_tail: List[int] = []
        self._limits: List[asyncio.Semaphore] = []
        self._next = 0

    def _connect(self) -> None:
        if self.teachers is None:
            if not self.teacher_names:
                raise RuntimeError("LookAheadOPD needs a teacher in explorer.auxiliary_models")
            self.teachers = [ray.get_actor(n, namespace=self.namespace) for n in self.teacher_names]
        if not self._limits:
            self._limits = [asyncio.Semaphore(self.parallel) for _ in self.teachers]

    async def _call(self, method: str, *args, **kwargs):
        """One teacher request, round-robin over the engines, bounded per engine."""
        i = self._next % len(self.teachers)
        self._next += 1
        async with self._limits[i]:
            for attempt in range(G.CALL_ATTEMPTS):
                try:
                    return await getattr(self.teachers[i], method).remote(*args, **kwargs)
                except Exception as e:  # engine hiccup: retry, then let the trainer stop loudly
                    if attempt == G.CALL_ATTEMPTS - 1:
                        raise
                    self.logger.warning(f"teacher {method} failed ({e!r}); retrying")
                    await asyncio.sleep(2.0)

    async def _teacher_marker(self) -> List[int]:
        """Tokens the teacher's chat template adds to the student's generation prompt, inserted between a
        turn's prompt and its response when the teacher scores it (e.g. Gemma 4 26B / 31B with thinking off
        end the prompt with an empty thought block that E2B's template lacks). Empty when the templates agree
        (e.g. Qwen3 -> Qwen3) or differ otherwise."""
        if self._marker is None:
            probe = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
            student = ray.get_actor(self.student_name, namespace=self.namespace)
            s_gen, s_bare, t_gen = await asyncio.gather(
                student.prompt_token_ids.remote(probe), student.prompt_token_ids.remote(probe, False),
                self.teachers[0].prompt_token_ids.remote(probe),
            )
            extra = len(t_gen) > len(s_gen) and t_gen[: len(s_gen)] == s_gen
            self._marker = t_gen[len(s_gen):] if extra else []
            # the student's generation prompt (e.g. `<|im_start|>assistant\n`): insert only after it
            self._prompt_tail = s_gen[len(s_bare):] if s_gen[: len(s_bare)] == s_bare else s_gen[-1:]
            self.logger.info(f"teacher scoring marker: {self._marker} after student prompt tail {self._prompt_tail}")
        return self._marker

    def _teacher_tokens(self, exp: Experience, marker: List[int]):
        """(tokens to score, index of the first response logprob) for one turn."""
        tokens, pl = exp.tokens.tolist(), exp.prompt_length
        tail = self._prompt_tail
        if marker and tail and tokens[pl - len(tail): pl] == tail:
            return tokens[:pl] + marker + tokens[pl:], pl + len(marker) - 1
        return tokens, pl - 1

    async def _ask(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int):
        responses = await self._call(
            "chat", messages, lora_request=None, temperature=temperature, max_tokens=max_tokens, n=1
        )
        text = responses[0].response_text or ""
        return G.parse_yes_no(text), text

    async def _lookahead(self, k: int, p: Dict, memory: List[Dict[str, str]]) -> Optional[Dict]:
        """(C) for one disagreed turn; None if it failed (the turn then counts as T)."""
        try:
            if self.lookahead_in_process:
                async def ask(messages, temperature, max_tokens):
                    responses = await self._call(
                        "chat", messages, lora_request=None, temperature=temperature, max_tokens=max_tokens, n=1
                    )
                    return responses[0].response_text or ""

                return await G.env_module(p["env"]).teacher_lookahead(
                    ask, p["task"], memory[: 2 * p["step"] + 1], p["prior_actions"], p["step"], p["gate"],
                    p["action"],
                )
            if self._lookahead_remote is None:
                # max_calls=1: a fresh process per look-ahead. Every ALFWorld game leaves a 35 MB planner
                # library mapped from TMPDIR (tmpfs) until its process exits (a ScienceWorld game: its Java
                # server).
                self._lookahead_remote = ray.remote(
                    num_cpus=1, max_calls=1, runtime_env={"env_vars": self._task_env}
                )(G.lookahead_task)
            return await self._lookahead_remote.remote(
                self.teacher_names[k % len(self.teacher_names)], self.namespace, p
            )
        except Exception as e:
            self.logger.warning(f"teacher look-ahead failed (step {p['step']} of {p['task']}): {e!r}")
            return None

    async def annotate(self, exps: List[Experience]) -> Dict[str, float]:
        """Gate and score the turns of a sampled batch in place; returns metrics."""
        turns = [e for e in exps if e.info and G.PAYLOAD_KEY in e.info]
        if len(turns) != len(exps):
            raise ValueError(f"{len(exps) - len(turns)} of {len(exps)} experiences carry no LookAheadOPD "
                             "payload (use a LookAheadOPD_*_workflow)")
        self._connect()
        t0 = time.time()
        payloads = [e.info[G.PAYLOAD_KEY] for e in turns]
        n = len(turns)
        memories = [G.payload_memory(p) for p in payloads]

        def ask_agree(i):
            g = payloads[i]["gate"]
            return self._ask(
                G.agree_messages(memories[i], payloads[i]["step"], payloads[i]["action"]),
                g["consistency_temperature"], g["consistency_max_tokens"],
            )

        def ask_progress(i):
            g, t = payloads[i]["gate"], payloads[i]["step"]
            return self._ask(
                G.progress_messages(memories[i], t, t + g["window_size"] - 1),
                g["progress_temperature"], g["progress_max_tokens"],
            )

        # (A) for every turn
        agree_answers = await asyncio.gather(*[ask_agree(i) for i in range(n)])
        t_agree = time.time() - t0
        disagreed = [i for i in range(n) if agree_answers[i][0] is not True]  # "No" or unparseable
        # (B) on the student's window, (C) the teacher's look-ahead: for the disagreed turns, together
        windowed = [i for i in disagreed if payloads[i]["has_window"]]
        t_ahead = time.time()
        progress_list, ahead_results = await asyncio.gather(
            asyncio.gather(*[ask_progress(i) for i in windowed]),
            asyncio.gather(*[self._lookahead(k, payloads[i], memories[i]) for k, i in enumerate(disagreed)]),
        )
        t_ahead = time.time() - t_ahead
        progress_answers = dict(zip(windowed, progress_list))
        lookaheads = dict(zip(disagreed, ahead_results))

        weights, classes = [], Counter()
        n_agree_bad = n_prog_bad = n_student_progress = n_teacher_progress = 0
        for i, exp in enumerate(turns):
            disagree = i in lookaheads
            student_np = teacher_p = None
            if disagree:
                if i in progress_answers:
                    says_progress = progress_answers[i][0]
                    n_prog_bad += says_progress is None
                    student_np = says_progress is not True  # "No" or unparseable -> not making progress
                else:  # final window: the episode outcome
                    student_np = not bool(payloads[i]["outcome"])
                result = lookaheads[i]
                teacher_p = result["progress"] if result is not None else None
                n_student_progress += student_np is False
                n_teacher_progress += teacher_p is not False  # unknown counts as progress
            n_agree_bad += agree_answers[i][0] is None
            weight = G.lookahead_weight(payloads[i]["gate"], disagree, student_np, teacher_p)
            weights.append(weight)
            classes[G.weight_class(disagree, student_np, teacher_p)] += 1
            if exp.metrics is None:
                exp.metrics = {}
            exp.metrics["opd_gate_disagree"] = float(disagree)
            exp.metrics["opd_gate_weight"] = weight
            if disagree:
                exp.metrics["opd_gate_not_progress"] = float(student_np)
                exp.metrics["opd_gate_teacher_no_progress"] = float(teacher_p is False)

        # 4. teacher log-probs of the student's tokens, blended by the weight
        t_prompts = time.time() - t0
        score = [i for i in range(n) if weights[i] != 0.0 or not self.skip_zero_weight_scoring]
        marker = await self._teacher_marker() if score else []
        inputs = {i: self._teacher_tokens(turns[i], marker) for i in score}
        scored = await asyncio.gather(*[
            self._call("logprobs", inputs[i][0], temperature=payloads[i]["gate"]["temperature"])
            for i in score
        ])
        teacher_logprobs = dict(zip(score, scored))
        raw_kl = []
        n_missing = n_marked = 0
        for i, exp in enumerate(turns):
            student = exp.logprobs
            valid = torch.full((len(student),), True, dtype=torch.bool)
            if i in teacher_logprobs:
                teacher = teacher_logprobs[i][inputs[i][1]:]
                assert len(teacher) == len(student), (
                    f"Length mismatch: teacher_logprobs={len(teacher)}, student_logprobs={len(student)}. "
                    f"tokens={len(exp.tokens)}, prompt_length={exp.prompt_length}"
                )
                n_marked += len(inputs[i][0]) != len(exp.tokens)
                # tokens the teacher engine returned no logprob for (vLLM.logprobs: NaN): no signal there
                missing = torch.isnan(teacher)
                if missing.any():
                    n_missing += int(missing.sum())
                    teacher = torch.where(missing, student, teacher)
                    valid &= ~missing
                raw_kl.append((student - teacher).sum().item())
                exp.teacher_logprobs = student + weights[i] * (teacher - student)
            else:  # weight 0, not scored: blending would give the student's own logprobs anyway
                exp.teacher_logprobs = student.clone()
            exp.teacher_logprobs_valid_mask = valid
            del exp.info[G.PAYLOAD_KEY]

        n_dis = len(disagreed)
        full = [payloads[i]["gate"]["both_weight"] for i in range(n)]
        metrics = {
            "time/teacher_gate": time.time() - t0,
            "time/teacher_gate_agree": t_agree,
            "time/teacher_gate_lookahead": t_ahead,  # (B) and (C)
            "time/teacher_gate_prompts": t_prompts,
            "time/teacher_gate_scoring": time.time() - t0 - t_prompts,
            "teacher_gate/turns": n,
            "teacher_gate/scored_turns": len(score),
            "teacher_gate/marked_turns": n_marked,  # scored with the teacher's generation-prompt marker
            "teacher_gate/missing_teacher_logprobs": n_missing,  # response tokens masked out (no teacher logprob)
            "teacher_gate/agree_prompts": n,
            "teacher_gate/progress_prompts": len(windowed),
            # mean weight over all turns, and the share of turns at the full weight / in between / at 0
            "teacher_gate/opd_gate_apply_rate": sum(weights) / n,
            "teacher_gate/opd_gate_full_rate": sum(w == f for w, f in zip(weights, full)) / n,
            "teacher_gate/opd_gate_half_rate": sum(0.0 < w < f for w, f in zip(weights, full)) / n,
            "teacher_gate/opd_gate_none_rate": sum(w == 0.0 for w in weights) / n,
            "teacher_gate/opd_gate_disagree_rate": n_dis / n,
            "teacher_gate/consistency_parse_success_rate": 1.0 - n_agree_bad / n,
        }
        if raw_kl:
            metrics["teacher_gate/raw_kl_per_turn"] = sum(raw_kl) / len(raw_kl)
        if windowed:
            metrics["teacher_gate/progress_parse_success_rate"] = 1.0 - n_prog_bad / len(windowed)
        if n_dis:
            # over the disagreed turns: who makes progress (B: answered or the outcome; C: unknown = progress),
            # and the share of each weight case
            metrics["teacher_gate/student_progress_rate"] = n_student_progress / n_dis
            metrics["teacher_gate/teacher_progress_rate"] = n_teacher_progress / n_dis
            metrics["teacher_gate/opd_gate_not_progress_rate"] = 1.0 - n_student_progress / n_dis
            for c in WEIGHT_CLASSES:
                metrics[f"teacher_gate/disagreed_{c}_rate"] = classes[c] / n_dis
            metrics.update(G.lookahead_metrics(list(lookaheads.values()), "teacher_gate/teacher_lookahead_"))
        return metrics
