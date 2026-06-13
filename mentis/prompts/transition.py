from __future__ import annotations

from typing import Any

from mentis.schema import MENTAL_TRANSITION_SCHEMA, PHYSICAL_TRANSITION_SCHEMA

from .common import prompt_header, prompt_json


def build_physical_transition_prompt(
    physical_state: dict[str, Any],
    mental_state: dict[str, Any],
    physical_action: dict[str, Any],
) -> str:
    return (
        prompt_header("Predict physical state transition for one candidate action")
        + "You are the physical state transition submodule of the Mental World Model. "
        "Compute s^{phy}_{t+1} from the physical action a^{phy}_t, the current physical "
        "state s^{phy}_t, and the current mental state s^{ment}_t. Use s^{ment}_t only as "
        "context for intention, attention, belief, or social constraints that affect what "
        "physical action is attempted; do not update mental_state here.\n"
        + "Transition procedure: identify the physical carrier in "
        "physical_action_description, check reachability, hand occupancy, affordance, "
        "visibility, occlusion, collision, contact, containment, support, temporal order, "
        "and environmental constraints, then update only objects, characters, relations, "
        "and environment facts that physically change. Keep relevant unchanged entities. "
        "If a^{phy}_t is empty, copy s^{phy}_t unless a communicative carrier such as "
        "speech or gesture physically occurs. Preserve every key in the template; use null "
        "for unavailable fields.\n"
        + "Output format: return exactly the strict physical_state schema shown below. "
        "This output is s^{phy}_{t+1}. The top-level keys must be only "
        "entity_and_attribute, relations, and environment. Do not return physical_state, "
        "mental_state, or any extra wrapper keys.\n"
        f"Strict local physical transition schema:\n{prompt_json(PHYSICAL_TRANSITION_SCHEMA)}\n"
        f"s^phy_t:\n{prompt_json(physical_state)}\n"
        f"s^ment_t:\n{prompt_json(mental_state)}\n"
        f"Physical action a^phy_t only:\n{prompt_json(physical_action)}"
    )


def build_mental_transition_prompt(
    physical_state: dict[str, Any],
    mental_state: dict[str, Any],
    action: dict[str, Any],
) -> str:
    return (
        prompt_header("Predict mental state transition for one candidate action")
        + "You are the mental state transition submodule of the Mental World Model. "
        "Compute s^{ment}_{t+1} from the physical action a^{phy}_t, the current physical "
        "state s^{phy}_t, the mental action a^{ment}_t, and the current mental state "
        "s^{ment}_t. This module runs in parallel with the physical transition submodule, "
        "so do not rely on s^{phy}_{t+1}; reason only from s^{phy}_t, s^{ment}_t, and the "
        "candidate action decomposition.\n"
        + "Transition procedure: update beliefs, attention, goals, intentions, emotions, "
        "preferences, norms, attitudes, role relations, and atmosphere when they are caused "
        "by the attempted physical carrier, explicit mental/social intent, speech content, "
        "gesture, deception, reassurance, persuasion, threat, cooperation, or norm violation. "
        "If a^{ment}_t is empty, still update mental_state only when the physical carrier "
        "would be perceived and naturally change beliefs or emotions; otherwise preserve "
        "s^{ment}_t. Preserve every key in the template; use null for unavailable fields.\n"
        + "Output format: return exactly the strict mental_state schema shown below. "
        "This output is s^{ment}_{t+1}. The top-level keys must be only "
        "mental_entity_and_attribute, relations, and atmosphere_of_environment. Do not "
        "return physical_state, mental_state, or any extra wrapper keys.\n"
        f"Strict local mental transition schema:\n{prompt_json(MENTAL_TRANSITION_SCHEMA)}\n"
        f"s^phy_t:\n{prompt_json(physical_state)}\n"
        f"s^ment_t:\n{prompt_json(mental_state)}\n"
        f"CandidateAction with a^phy_t and a^ment_t:\n{prompt_json(action)}"
    )
