from __future__ import annotations

from typing import Any

from mentis.schema import TARGET_OBSERVATION_TEMPLATE

from .common import prompt_header, prompt_json


def build_observation_generation_prompt(
    world_state: dict[str, Any], target_agent: str
) -> str:
    return (
        prompt_header("Render first-person target observation from global WorldState")
        + "You are the world model observation-generation module. Input is the current strict "
        "WorldState s_t and the dataset's target agent description. Return the target agent's "
        "observation o_t from that agent's perspective.\n"
        + "physical_observation must use the same schema as physical_state, but include only "
        "entities, relations, and environment facts the target can see, hear, touch, locate, "
        "or otherwise directly perceive at the current moment. Preserve keys; put null for "
        "inaccessible scalar fields, and use empty or null-filled lists for inaccessible "
        "list entries.\n"
        + "mental_observation must use the same schema as mental_state. Include the target "
        "agent's own beliefs, goals, intentions, emotions, preferences, and norms, plus only "
        "cue-supported inferences about other agents. Do not copy hidden global mental state "
        "as mind reading.\n"
        + "Preserve every key in the template, do not add extra keys, and use null for "
        "unavailable fields. Use only the top-level keys physical_observation and "
        "mental_observation. Do not return target_agent_id, target_agent_name, or an "
        "observation wrapper object.\n"
        f"Strict local observation template:\n{prompt_json(TARGET_OBSERVATION_TEMPLATE)}\n"
        f"Dataset target agent description: {target_agent}\n"
        f"Current WorldState s_t:\n{prompt_json(world_state)}"
    )
