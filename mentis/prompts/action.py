from __future__ import annotations

from .common import prompt_header, prompt_json


def build_action_parser_prompt(action_description: str) -> str:
    output_schema = {
        "physical_action_description": (
            "a_t^phy: physical execution/carrier/direct physical effect, or empty string"
        ),
        "mental_action_description": (
            "a_t^ment: semantic, intention, emotion, cognitive, or social component, or empty string"
        ),
    }
    return (
        prompt_header("Parse one candidate action into physical and mental dimensions")
        + "Input is exactly one natural-language candidate action a_t. Use only this action "
        "text. Do not use target-agent identity, question text, observation, current state, "
        "or option correctness. Do not add, drop, merge, rank, or rewrite actions.\n"
        + "Decompose a_t according to the Chapter 3 CandidateAction definition:\n"
        + "- a_t^phy is the physical execution, physical carrier, or direct physical-world "
        "effect of the action. It includes bodily movement, object manipulation, speech "
        "as sound production, gaze, facial expression, gesture, touching, moving, handing "
        "an object, pointing, nodding, shaking the head, giving a thumbs-up, making a "
        "timeout gesture, or signaling a pass.\n"
        + "- a_t^ment is the semantic, intention, emotion, cognitive-control, or social "
        "component carried by the physical action. It includes comforting, requesting, "
        "threatening, deceiving, persuading, reassuring, misleading, expressing "
        "friendliness, expressing anger, asking for help, giving permission, refusing, "
        "or trying to update someone's belief or goal.\n"
        + "Return JSON keys physical_action_description and mental_action_description. "
        "If a dimension is not explicitly present in the action text, return an empty "
        "string for that dimension. A simple gesture such as \"waves\" has a_t^phy "
        "but empty a_t^ment unless the text says it is greeting, comforting, warning, "
        "or otherwise socially/mentally meaningful. Speech has a physical carrier "
        "because the agent speaks, and the spoken content usually supplies a_t^ment.\n"
        + "Examples:\n"
        + "1. Action: \"waves\" -> a_t^phy: \"waves hand\"; a_t^ment: \"\".\n"
        + "2. Action: \"waves to greet the class\" -> a_t^phy: \"waves hand\"; "
        "a_t^ment: \"expresses greeting and friendliness to the class\".\n"
        + "3. Action: \"says 'Don't worry'\" -> a_t^phy: \"speaks the sentence "
        "'Don't worry'\"; a_t^ment: \"reassures the listener\".\n"
        + "4. Action: \"hands over the folder\" -> a_t^phy: \"hands over the folder\"; "
        "a_t^ment: \"\".\n"
        + "5. Action: \"points to the exit and tells him to leave\" -> a_t^phy: "
        "\"points to the exit and speaks\"; a_t^ment: \"directs him to leave\".\n"
        f"Output schema:\n{prompt_json(output_schema)}\n"
        f"Action a_t:\n{action_description}"
    )
