from __future__ import annotations

from typing import Any

from mentis.schema import WORLD_STATE_TEMPLATE

from .common import prompt_header, prompt_json


def build_state_parser_prompt(sample: dict[str, Any]) -> str:
    return (
        prompt_header("Parse multimodal scene into joint physical-mental WorldState")
        + "You must parse the scene at the current moment into s_t. Replace null placeholders "
        "with observed or reasonably inferred content. If a field cannot be determined from "
        "text/images/video, keep that field as null. Preserve every key in the template and "
        "do not add extra keys. Template arrays are examples of item shape only; arrays may "
        "contain as many observed objects, characters, individuals, relations, attitudes, "
        "or role relations as the current scene supports. Use only story.text, story.images, "
        "and story.video as input; "
        "do not use target agent, question, answer options, candidate actions, golden answer, "
        "or any future-branch information when constructing s_t.\n"
        + "Use the exact top-level keys physical_state and mental_state. Physical state must "
        "describe objects, characters, spatial/contact relations, and environment. Mental "
        "state must describe each character's beliefs, attention, goals, intentions, emotions, "
        "preferences, norms, attitudes, role relations, and atmosphere. Use null for unknown "
        "semantic_content, and use concise strings in all lists.\n"
        f"Strict local state template:\n{prompt_json(WORLD_STATE_TEMPLATE)}\n"
        f"Current story-only input:\n{prompt_json(sample)}"
    )
