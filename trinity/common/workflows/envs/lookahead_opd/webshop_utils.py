# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified for DASH-OPD, 2026 (WebShop prompts of arXiv 2607.29078), and for LookAheadOPD.

import re
from typing import Sequence

# --------------------- WebShop --------------------- #
WEBSHOP_TEMPLATE_NO_HIS = """
You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason about the current situation, then think which admissible action best advances the shopping goal.
Once you've finished your reasoning, you should choose the best admissible action for the current step and present it within <action> </action> tags.
Do not output any other text besides your reasoning and the action.
"""

WEBSHOP_TEMPLATE = """
You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason about the current situation, then think which admissible action best advances the shopping goal.
Once you've finished your reasoning, you should choose the best admissible action for the current step and present it within <action> </action> tags.
Do not output any other text besides your reasoning and the action.
"""

HISTORY_LENGTH = 2
INITIAL_NO_HISTORY_TURNS = 3  # DASH-OPD's WebShop configs
MEMORY_FORMAT = "[Observation {step_num}: '{obs}', Action {step_num}: '{act}']"


def build_webshop_prompt(
    *,
    task_description: str,
    step_count: int,
    history: Sequence[str],
    history_length: int,
    initial_no_history_turns: int = INITIAL_NO_HISTORY_TURNS,
    current_observation: str,
    available_actions: str,
    use_initial_no_history_template: bool = False,
) -> str:
    """Build a WebShop prompt under the shared explicit history policy."""

    if history_length < -1:
        raise ValueError(f"history_length must be -1 or non-negative, got {history_length}")
    if isinstance(initial_no_history_turns, bool) or not isinstance(initial_no_history_turns, int):
        raise TypeError(
            "initial_no_history_turns must be an integer, got "
            f"{type(initial_no_history_turns).__name__}"
        )
    if initial_no_history_turns < 0:
        raise ValueError(
            f"initial_no_history_turns must be non-negative, got {initial_no_history_turns}"
        )

    if use_initial_no_history_template and step_count < initial_no_history_turns:
        return WEBSHOP_TEMPLATE_NO_HIS.format(
            task_description=task_description,
            current_observation=current_observation,
            available_actions=available_actions,
        )

    selected_history = list(history) if history_length == -1 else list(history[-history_length:])
    return WEBSHOP_TEMPLATE.format(
        task_description=task_description,
        step_count=step_count,
        history_length=len(selected_history),
        action_history="\n".join(selected_history),
        current_step=step_count + 1,
        current_observation=current_observation,
        available_actions=available_actions,
    )


def parse_action(response: str) -> str:
    try:
        return response.split("<action>")[1].split("</action>")[0].strip()
    except Exception as e:
        print(f"Error parsing action: {e}, response = {response}")
        return ""


def validate_action(action: str, available_actions: dict):
    pattern = re.compile(r"(.+)\[(.+)\]")
    match = re.match(pattern, action)
    if match is None:
        action_name = action
        action_arg = None
    else:
        action_name, action_arg = match.groups()

    if action_arg is not None:
        action_arg = action_arg.lower()

    if (
        action_name == "search"
        and action_arg is not None
        and action_arg != ""
        and available_actions.get("has_search_bar", False)
    ):
        return True, ""
    if action_name == "click" and action_arg in available_actions.get("clickables", []):
        return True, ""

    if action_name == "search":
        if action_arg == "" or action_arg is None:
            return (
                False,
                "Invalid action, please type in the query you want to search in the square brackets here.",
            )
        return (
            False,
            "Can not perform search action without search bar. Please click the Back to Search button first.",
        )

    if action_name == "click" and action_arg not in available_actions.get("clickables", []):
        return (
            False,
            "Incorrect action format, make sure you have the correct button name that is "
            f"within the current page. The buttons you can click now are: {available_actions.get('clickables', [])}",
        )

    return (
        False,
        "Invalid action. You should wrap your action with the <action> </action> tag and follow the action format.",
    )


def format_observation(observation: str) -> str:
    return observation.strip()


def _compact_text(text: str) -> str:
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def _extract_task_description(observation: str) -> str:
    """Extract the instruction text from a WebShop observation."""
    lines = [line.strip() for line in observation.splitlines()]
    try:
        instruction_idx = lines.index("Instruction:")
    except ValueError:
        return _compact_text(observation)

    task_lines = []
    for line in lines[instruction_idx + 1 :]:
        if not line:
            continue
        if line.startswith("[button]"):
            break
        task_lines.append(line)

    return " ".join(task_lines).strip() or _compact_text(observation)


def _format_history(observation: str, step_num: int, action: str) -> str:
    return MEMORY_FORMAT.format(step_num=step_num, obs=_compact_text(observation), act=action)


def _format_available_actions(available_actions: dict) -> str:
    formatted_actions = []
    if available_actions.get("has_search_bar", False):
        formatted_actions.append("search[<query>]")
    for clickable in available_actions.get("clickables", []):
        formatted_actions.append(f"click[{clickable}]")
    return "\n".join(formatted_actions)
