"""LookAheadOPD: on-policy distillation gated by the teacher's agreement and by student and teacher look-aheads.

gate.py          the method: teacher prompts, answer parsing, weights, per-turn payload, look-ahead Ray task
workflow.py      explorer side: the student's episode, a payload on every turn (no teacher call)
alfworld.py, scienceworld.py, webshop.py   the three environments (rollout + the teacher's look-ahead)
*_utils.py, webshop_env_server.py          prompts and game helpers

The trainer side is trinity/trainer/tunix/teacher_gate.py.
"""
