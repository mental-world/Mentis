from __future__ import annotations

import json
from typing import Any, Optional

from .schema import (
    MENTAL_STATE_TEMPLATE,
    OBSERVATION_TEMPLATE,
    PHYSICAL_STATE_TEMPLATE,
    WORLD_STATE_TEMPLATE,
)


def as_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def header(task: str) -> str:
    return f"Task: {task}\nReturn only valid JSON. Do not use markdown. Keep reasoning concise.\n"


MODALITY_GUIDANCE = {
    "text": (
        "Input modality: TEXT. story.text is the complete narrative; every physical fact and "
        "every observable mental cue (behavior, speech, gaze, expression) is in the prose. "
        "Parse states from what the text shows, not from your own additions.\n"
    ),
    "image": (
        "Input modality: IMAGE SEQUENCE. The story is carried entirely by the ordered images; "
        "scene_context only anchors identities (who is who by appearance/props). "
        "Caption boxes rendered inside the images are the narration - read them verbatim as "
        "story events and quoted speech. Images are in temporal order: a later image depicts a "
        "later moment; track object positions, possession, occlusion, and each person's gaze "
        "and posture ACROSS images, and note which characters could or could not see each "
        "depicted event (visibility asymmetries are usually the point of the scene).\n"
    ),
    "video": (
        "Input modality: VIDEO. You receive frames sampled in temporal order (timestamps "
        "given) and, when available, an audio transcript of spoken dialogue and salient "
        "sounds. Treat transcript lines as in-world speech: attribute each line to its "
        "speaker by matching the frames (who is talking, to whom, who is within earshot). "
        "scene_context only anchors identities. Reconstruct the event order across frames, "
        "track object states and who-saw/heard-what, and remember that characters looking "
        "away or out of earshot at a moment do not know what happened then.\n"
    ),
}


def state_prompt(story_block: dict[str, Any], modality: str) -> str:
    guidance = MODALITY_GUIDANCE.get(modality, MODALITY_GUIDANCE["text"])
    return (
        header("Parse the scene into a joint physical-mental world state")
        + guidance
        + "Parse the scene at the current moment into the world state s_t. Replace null "
        "placeholders with observed or reasonably inferred content. If a field cannot be "
        "determined from the input, keep that field as null. Preserve every key in the "
        "template and do not add extra keys. Template arrays are examples of item shape "
        "only; arrays may contain as many observed objects, characters, individuals, "
        "relations, attitudes, or role relations as the current scene supports. Use only "
        "the story input; do not use any question or answer options when constructing s_t.\n"
        + "Use the exact top-level keys physical_state and mental_state. Physical state must "
        "describe objects, characters, spatial/contact relations, and environment. Mental "
        "state must describe each character's beliefs, attention, goals, intentions, "
        "emotions, preferences, norms, attitudes, role relations, and atmosphere. Use null "
        "for unknown semantic_content, and use concise strings in all lists.\n"
        f"Strict state template:\n{as_json(WORLD_STATE_TEMPLATE)}\n"
        f"Story input:\n{as_json(story_block)}"
    )


def observation_prompt(world_state: dict[str, Any], target_agent: str) -> str:
    return (
        header("Render the target agent's first-person observation from the world state")
        + "Input is the current world state s_t and the target agent description. Return the "
        "target agent's observation o_t from that agent's perspective.\n"
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
        "mental_observation.\n"
        f"Strict observation template:\n{as_json(OBSERVATION_TEMPLATE)}\n"
        f"Target agent description: {target_agent}\n"
        f"Current world state s_t:\n{as_json(world_state)}"
    )


ACTION_RULES = (
    "Decompose each action into two components:\n"
    "- physical_action is the physical execution, physical carrier, or direct physical-world "
    "effect of the action. It includes bodily movement, object manipulation, speech as sound "
    "production, gaze, facial expression, gesture, touching, moving, handing an object, "
    "pointing, nodding, shaking the head, or signaling.\n"
    "- mental_action is the semantic, intention, emotion, cognitive, or social component "
    "carried by the physical action. It includes comforting, requesting, threatening, "
    "deceiving, persuading, reassuring, misleading, expressing friendliness, expressing "
    "anger, asking for help, giving permission, refusing, or trying to update someone's "
    "belief or goal.\n"
    "If a component is not explicitly present in the action text, return an empty string for "
    "it. Speech has a physical carrier because the agent speaks, and the spoken content "
    "usually supplies the mental component.\n"
)


def actions_prompt(options: list[dict[str, Any]]) -> str:
    output_schema = {
        "actions": [
            {
                "option_id": "A",
                "physical_action": "physical execution/carrier, or empty string",
                "mental_action": "intended cognitive-social effect, or empty string",
            }
        ]
    }
    return (
        header("Decompose candidate actions into physical and mental components")
        + "Input is a list of natural-language candidate actions. Use only each action text. "
        "Do not use the question, the scene, option correctness, or cross-option comparison. "
        "Preserve every option_id exactly and return one item per input action.\n"
        + ACTION_RULES
        + "Do not add, drop, merge, rank, or rewrite actions.\n"
        f"Output schema:\n{as_json(output_schema)}\n"
        f"Actions:\n{as_json(options)}"
    )


def single_action_prompt(action_description: str) -> str:
    output_schema = {
        "physical_action": "physical execution/carrier, or empty string",
        "mental_action": "intended cognitive-social effect, or empty string",
    }
    return (
        header("Decompose one candidate action into physical and mental components")
        + "Input is exactly one natural-language candidate action. Use only this action "
        "text.\n"
        + ACTION_RULES
        + "Examples:\n"
        + "1. \"waves\" -> physical_action: \"waves hand\"; mental_action: \"\".\n"
        + "2. \"waves to greet the class\" -> physical_action: \"waves hand\"; "
        "mental_action: \"expresses greeting and friendliness to the class\".\n"
        + "3. \"says 'Don't worry'\" -> physical_action: \"speaks the sentence 'Don't "
        "worry'\"; mental_action: \"reassures the listener\".\n"
        f"Output schema:\n{as_json(output_schema)}\n"
        f"Action:\n{action_description}"
    )


def physical_transition_prompt(
    physical_state: dict[str, Any],
    mental_state: dict[str, Any],
    action: dict[str, Any],
) -> str:
    return (
        header("Predict the physical state transition for one candidate action")
        + "You are the physical transition submodule of a mental world model. Compute the "
        "next physical state from the physical component of the action, the current "
        "physical state, and the current mental state. Use the mental state only as context "
        "for intention, attention, belief, or social constraints that affect what physical "
        "action is attempted; do not update the mental state here.\n"
        + "Procedure: identify the physical carrier in physical_action, check reachability, "
        "hand occupancy, affordance, visibility, occlusion, collision, contact, containment, "
        "support, temporal order, and environmental constraints, then update only objects, "
        "characters, relations, and environment facts that physically change. Keep relevant "
        "unchanged entities. If physical_action is empty, copy the current physical state "
        "unless a communicative carrier such as speech or gesture physically occurs. "
        "Preserve every key in the template; use null for unavailable fields.\n"
        + "Return exactly the physical-state schema below; the top-level keys must be only "
        "entity_and_attribute, relations, and environment. Do not return any wrapper keys.\n"
        f"Physical-state schema:\n{as_json(PHYSICAL_STATE_TEMPLATE)}\n"
        f"Current physical state:\n{as_json(physical_state)}\n"
        f"Current mental state:\n{as_json(mental_state)}\n"
        f"Physical component of the action:\n{as_json(action)}"
    )


def mental_transition_prompt(
    physical_state: dict[str, Any],
    mental_state: dict[str, Any],
    action: dict[str, Any],
    next_physical_state: dict[str, Any],
) -> str:
    return (
        header("Predict the mental state transition for one candidate action")
        + "You are the mental transition submodule of a mental world model. Compute the next "
        "mental state from the current joint state, the candidate action (physical and "
        "mental components), and the already predicted next physical state. Treat the "
        "physical future as the perceived/audible/visible carrier through which mental "
        "effects happen. Mental updates must be causally grounded in the action and in the "
        "next physical state; do not invent psychological changes unsupported by the "
        "physical carrier, speech content, gesture, visibility, audibility, social relation, "
        "or scene context.\n"
        + "Procedure: update beliefs, attention, goals, intentions, emotions, preferences, "
        "norms, attitudes, role relations, and atmosphere when they are caused by the "
        "attempted physical carrier, explicit mental/social intent, speech content, gesture, "
        "deception, reassurance, persuasion, threat, cooperation, or norm violation. If "
        "mental_action is empty, still update the mental state only when the physical "
        "carrier would be perceived and naturally change beliefs or emotions; otherwise "
        "preserve the current mental state. Preserve every key in the template; use null "
        "for unavailable fields.\n"
        + "Return exactly the mental-state schema below; the top-level keys must be only "
        "mental_entity_and_attribute, relations, and atmosphere_of_environment. Do not "
        "return any wrapper keys.\n"
        f"Mental-state schema:\n{as_json(MENTAL_STATE_TEMPLATE)}\n"
        f"Current physical state:\n{as_json(physical_state)}\n"
        f"Predicted next physical state:\n{as_json(next_physical_state)}\n"
        f"Current mental state:\n{as_json(mental_state)}\n"
        f"Candidate action with physical_action and mental_action:\n{as_json(action)}"
    )


GRADE_MAPPING = (
    "For each numeric dimension, first assign an internal 1-5 grade, then output only the "
    "scaled [0,1] score using this mapping: grade 1 -> 0.0, grade 2 -> 0.25, grade 3 -> 0.5, "
    "grade 4 -> 0.75, grade 5 -> 1.0. Do not add grade keys.\n"
)

DIMENSION_DEFS = (
    "mentally_consistent evaluates whether the action follows from the target agent's "
    "evidenced mental state under o_t: beliefs, attention, goals, intentions, emotions, "
    "preferences, personality, and perspective. Score consistency with the mindset, not "
    "whether the outcome succeeds. "
    "physically_plausible evaluates whether the action and imagined next state are "
    "physically possible from s_t. "
    "socially_appropriate evaluates custom, moral, politeness, role, and social norm "
    "appropriateness.\n"
)

RUBRICS = (
    "physically_plausible rubric: grade 5 means fully physically possible from s_t; grade 3 "
    "means partially plausible or uncertain; grade 1 means serious physical impossibility or "
    "contradiction in reachability, affordance, visibility, contact, support, collision, "
    "containment, temporal order, or causal transition.\n"
    "mentally_consistent rubric (mindset fit, not outcome success): grade 5 means the branch "
    "follows naturally from the target's evidenced beliefs, attention, goals, intentions, "
    "emotions, preferences, personality, and perspective under o_t; grade 3 means mixed or "
    "underspecified; grade 1 means clear contradiction with the question, o_t, or target "
    "mindset.\n"
    "socially_appropriate rubric: grade 5 means fully appropriate under norms, roles, "
    "morality, politeness, customs, and relationship context; grade 3 means socially mixed; "
    "grade 1 means obvious social, moral, role, politeness, or custom violation.\n"
)

COMPARATIVE_DISCIPLINE = (
    "Comparative discipline: wrong actions usually violate at least one constraint "
    "inferable from s_t/o_t (someone's belief or perceptual access, an object's state or "
    "reach, a stated agreement or norm, timing). Before grading, check each branch for such "
    "violations; a violated constraint caps that dimension at grade 3. Reserve grade 5 in a "
    "dimension for branches with no identifiable flaw in it; do not give two branches "
    "identical grades on every dimension unless they are genuinely indistinguishable after "
    "this check.\n"
    "Additionally output relative_rank for every branch: a strict ranking 1..N (1 = best "
    "overall fit with the target's evidenced mental state and the situation; no ties, every "
    "integer used exactly once).\n"
)


def scores_prompt(
    target_agent: str,
    world_state: dict[str, Any],
    observation: dict[str, Any],
    question: str,
    branches: list[dict[str, Any]],
) -> str:
    output_schema = {
        "scores": [
            {
                "option_id": "A",
                "mentally_consistent": "0.0",
                "physically_plausible": "0.0",
                "socially_appropriate": "0.0",
                "safety_veto": "true/false",
                "relative_rank": "1..N unique",
                "reasoning": "concise rationale",
            }
        ]
    }
    return (
        header("Evaluate imagined action branches")
        + "Use the full tuple (target agent description, question, s_t, o_t, action, next "
        "state) to score every action branch. Return exactly one score object per input "
        "option_id. Do not choose the final answer and do not add extra keys.\n"
        + "Scores must be strings or numbers in [0,1]. "
        + DIMENSION_DEFS
        + GRADE_MAPPING
        + RUBRICS
        + "safety_veto must be true only for illegal actions or serious safety risks; "
        "otherwise false. Ground every score in the s_t -> action -> next-state branch, not "
        "option wording alone. Each reasoning must be concise and cite branch-specific "
        "evidence.\n"
        + COMPARATIVE_DISCIPLINE
        + f"Output schema:\n{as_json(output_schema)}\n"
        f"Target agent description:\n{target_agent}\n"
        f"Question:\n{question}\n"
        f"s_t:\n{as_json(world_state)}\n"
        f"o_t:\n{as_json(observation)}\n"
        f"Branches:\n{as_json(branches)}"
    )


def direct_prompt(
    modality: str,
    scene_block: dict[str, Any],
    question: str,
    options: list[dict[str, Any]],
) -> str:
    output_schema = {"choice": "option_id"}
    parts = [
        f"The current scene is presented in {modality} modality. "
        "Choose the most appropriate option based on the scene and question. "
        "Return only valid JSON.",
        f"Output schema:\n{as_json(output_schema)}",
        "Return exactly one JSON key: choice. Do not add reason, confidence, or any other keys.",
    ]
    if scene_block:
        parts.append(f"Scene:\n{as_json(scene_block)}")
    parts.append(f"Question:\n{question}")
    parts.append(f"Options:\n{as_json(options)}")
    return "\n".join(parts)
