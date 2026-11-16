"""ScienceWorld prompts and game helpers (TCOD's ScienceWorld setup, with DASH-OPD's prompts, arXiv 2607.29078
Appendix L)."""

import json
from typing import Any, Dict, List, Tuple

# The first NO_HISTORY_TURNS turns have no history summary; from then on the last HISTORY_LENGTH
# observation-action pairs.
DASH_SCIWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ScienceWorld text environment.
{task_description}
Your current observation is: {current_observation}
Available action commands: [{action_templates}]
Available objects you can interact with: [{objects}]

Now it's your turn to take an action. Combine an action command with appropriate object(s) to form a valid action.
You should first reason about the current situation.
Once you've finished your reasoning, you should choose the best valid action for the current step and present it within <action> </action> tags.
Do not output any other text besides your reasoning and the action.
"""

DASH_SCIWORLD_TEMPLATE = """
You are an expert agent operating in the ScienceWorld text environment.
{task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Available action commands: [{action_templates}]
Available objects you can interact with: [{objects}]

Now it's your turn to take an action. Combine an action command with appropriate object(s) to form a valid action.
You should first reason about the current situation.
Once you've finished your reasoning, you should choose the best valid action for the current step and present it within <action> </action> tags.
Do not output any other text besides your reasoning and the action.
"""

HISTORY_LENGTH = 2
NO_HISTORY_TURNS = 3
MEMORY_FORMAT = "[Observation {step_num}: '{obs}', Action {step_num}: '{act}']"


def parse_action(response: str) -> str:
    try:
        return response.split("<action>")[1].split("</action>")[0].strip()
    except Exception as e:
        print(f"Error parsing action: {e}, response = {response}")
        return ""


def format_observation(observation: str) -> str:
    return observation.strip()


def _extract_task(task_description: str) -> str:
    return task_description.strip()


def _format_history(observation: str, step_num: int, act: str) -> str:
    return MEMORY_FORMAT.format(step_num=step_num, obs=observation.strip(), act=act)


def _parse_task_config(task_desc: Any) -> Dict[str, Any]:
    if isinstance(task_desc, dict):
        return task_desc
    if isinstance(task_desc, str):
        return json.loads(task_desc)
    raise TypeError(f"Unsupported ScienceWorld task description type: {type(task_desc)}")


def _get_compact_action_info(env) -> Tuple[List[str], List[str]]:
    """Action templates (~23) and objects (~20) of the current state, instead of the ~1162 full
    action-object combinations of get_valid_action_object_combinations()."""
    try:
        actions = env.get_possible_actions()
    except AttributeError:
        actions = env.getPossibleActions()

    try:
        objects = env.get_possible_objects()
    except AttributeError:
        objects = env.getPossibleObjects()

    if isinstance(actions, str):
        actions = [actions] if actions else []
    if isinstance(objects, str):
        objects = [objects] if objects else []

    return actions, objects


def _create_scienceworld_env(
    task_desc: Any,
    *,
    max_env_steps: int = 30,
    generate_gold_path: bool = False,
):
    task_config = _parse_task_config(task_desc)

    try:
        from scienceworld import ScienceWorldEnv
    except Exception as e:
        raise ImportError(
            f"Error importing ScienceWorldEnv: {e}. "
            "Please make sure the scienceworld package is installed successfully: "
            "https://github.com/allenai/ScienceWorld"
        ) from e

    task_name = task_config["task_name"]
    var_num = task_config["var_num"]
    jar_path = task_config.get("jar_path", "")  # empty: the jar bundled with the scienceworld package
    simplification_str = task_config.get("simplification_str", "easy")
    env_step_limit = max(task_config.get("env_step_limit", 100), max_env_steps + 5)

    env = ScienceWorldEnv("", jar_path, envStepLimit=env_step_limit)
    env.load(task_name, var_num, simplification_str, generateGoldPath=generate_gold_path)
    return env


def _reset_scienceworld_env(env) -> Tuple[str, Dict[str, Any], str]:
    observation, info = env.reset()
    task_description = _extract_task(env.get_task_description())
    return observation, info, task_description


def build_user_content(env, observation: str, history: List[str], task_description: str, r: int):
    """The user message of step `r` (0-indexed) and the formatted observation."""
    format_obs = format_observation(observation)
    action_templates, objects = _get_compact_action_info(env)
    reformatted_actions = ", ".join(f"'{s}'" for s in action_templates if s != "help")
    reformatted_objects = ", ".join(f"'{s}'" for s in objects)
    if len(history) < NO_HISTORY_TURNS:
        user_content = DASH_SCIWORLD_TEMPLATE_NO_HIS.format(
            task_description=task_description,
            current_observation=format_obs,
            action_templates=reformatted_actions,
            objects=reformatted_objects,
        )
    else:
        user_content = DASH_SCIWORLD_TEMPLATE.format(
            task_description=task_description,
            step_count=r,
            history_length=min(HISTORY_LENGTH, len(history)),
            action_history="\n".join(history[-HISTORY_LENGTH:]),
            current_step=r + 1,
            current_observation=format_obs,
            action_templates=reformatted_actions,
            objects=reformatted_objects,
        )
    return user_content, format_obs


def replay_game(task_desc: str, actions: List[str], step: int):
    """A fresh copy of the game after the first `step` of `actions`, played exactly as the student's game:
    created with generate_gold_path=False, and with the action and object lists requested before every
    step (the student's rollout asks for them every turn; ScienceWorld names objects differently later if
    they are not asked). Returns (env, observation, info, history, task_description, n_replayed, done)."""
    env = _create_scienceworld_env(task_desc, generate_gold_path=False)
    observation, info, task_description = _reset_scienceworld_env(env)
    history: List[str] = []
    done = False
    for k, action in enumerate(actions[:step]):
        _get_compact_action_info(env)
        history.append(_format_history(format_observation(observation), k + 1, action))
        observation, _, done, info = env.step(action)
        if done:
            break
    return env, observation, info, history, task_description, len(history), done
