from __future__ import annotations

from typing import Any

from .common import prompt_json


def build_direct_answer_baseline_prompt(sample: dict[str, Any]) -> str:
    output_schema = {"final_action": "option_id"}
    modality = str(sample.get("modality") or "unknown")
    parts = (
        f"The current scene is presented in {modality} modality. "
        "Choose the most appropriate option based on the scene and question.",
        f"Output schema:\n{prompt_json(output_schema)}",
        "Return exactly one JSON key: final_action. Do not add reason, confidence, or any other keys.",
    )
    prompt_parts = list(parts)
    if modality == "text":
        prompt_parts.append(f"Scene:\n{prompt_json(sample.get('scene', {}))}")
    prompt_parts.extend(
        [
            f"Question:\n{sample.get('question', '')}",
            f"Options:\n{prompt_json(sample.get('options', []))}",
        ]
    )
    return "\n".join(prompt_parts)
