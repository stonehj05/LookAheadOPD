"""One ALFWorld eval episode with the full conversation as memory (every turn's user and assistant messages
are kept and resent).

- No action validation: whatever is parsed is passed to env.step, as in training. `action_admissible` is
  recorded per step as a diagnostic only.
- Success is `done` (ALFWorld signals a win with done=True), as training's reward.
- The prompt switches to the with-history template from turn 4 (r = 3, 0-indexed), the DASH-OPD paper's eval
  protocol (training switches from turn 3; see lookahead_opd/alfworld.py). This is the protocol of the
  paper's ALFWorld numbers.
"""

from typing import Dict, List, Optional

from alfworld_agent_utils import (
    HISTORY_LENGTH,
    ALFWORLD_TEMPLATE,
    ALFWORLD_TEMPLATE_NO_HIS,
    _format_admissible_commands,
    _format_history,
    format_observation,
    parse_action,
)

NO_HISTORY_TURNS = HISTORY_LENGTH + 1


def run_episode(
    env,
    model_client,
    task_description: str,
    observation: str,
    info: Dict,
    history: Optional[List[str]] = None,
    start_step: int = 0,
    max_steps: int = 30,
) -> Dict:
    history = list(history) if history else []
    step_log: List[Dict] = []
    actions: List[str] = []
    memory: List[Dict[str, str]] = []

    done = False
    final_reward = 0.0
    last_step_index = start_step - 1

    for r in range(start_step, start_step + max_steps):
        last_step_index = r
        formatted_observation = format_observation(observation)
        admissible_commands = info.get("admissible_commands", [])
        if admissible_commands and isinstance(admissible_commands[0], list):
            admissible_commands = admissible_commands[0]
        formatted_actions = _format_admissible_commands(info)

        if len(history) < NO_HISTORY_TURNS:
            user_content = ALFWORLD_TEMPLATE_NO_HIS.format(
                current_observation=formatted_observation,
                admissible_actions=formatted_actions,
            )
        else:
            user_content = ALFWORLD_TEMPLATE.format(
                task_description=task_description,
                step_count=r,
                history_length=min(HISTORY_LENGTH, len(history)),
                action_history="\n".join(history[-HISTORY_LENGTH:]),
                current_step=r + 1,
                current_observation=formatted_observation,
                admissible_actions=formatted_actions,
            )

        memory = memory + [{"role": "user", "content": user_content}]
        response_text, raw = model_client.chat(
            memory, available_actions={"commands": admissible_commands}
        )
        memory.append({"role": "assistant", "content": response_text})
        finish_reason = None
        if raw and raw.get("choices"):
            finish_reason = raw["choices"][0].get("finish_reason")

        action = parse_action(response_text)
        history.append(_format_history(formatted_observation, r + 1, action))
        actions.append(action)

        observation, reward, done, info = env.step(action)

        step_log.append(
            {
                "step": r,
                "prompt": user_content,
                "response": response_text,
                "finish_reason": finish_reason,
                "parsed_action": action,
                "action_admissible": action in admissible_commands,
                "reward": reward,
                "done": done,
            }
        )

        if done:
            final_reward = 1.0
            break

    return {
        "actions": actions,
        "env_rounds": last_step_index - start_step + 1,
        "final_reward": final_reward,
        "done": done,
        "step_log": step_log,
        "final_observation": observation,
        "final_info": info,
    }
