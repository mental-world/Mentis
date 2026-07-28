from __future__ import annotations

from typing import Any

from . import prompts
from .engine import RunStats, public_record
from .llm import LLMClient
from .media import scene_from_story
from .schema import DirectChoice, Sample


async def run_direct_sample(
    llm: LLMClient, record: dict[str, Any], base_dir: str = ""
) -> dict[str, Any]:
    output = public_record(record)
    stats = RunStats()
    try:
        sample = Sample.model_validate(record)
        scene = scene_from_story(sample.story.as_dict(), base_dir)
        scene_block: dict[str, Any] = {}
        if scene["modality"] == "text":
            scene_block["text"] = scene["text"]
        elif scene.get("scene_context"):
            scene_block["scene_context"] = scene["scene_context"]
        options = [
            {"option_id": o.option_id, "action_description": o.action_description}
            for o in sample.options
        ]
        reply = await llm.json_call(
            prompt=prompts.direct_prompt(scene["modality"], scene_block, sample.question, options),
            schema=DirectChoice,
            scene=scene,
        )
        stats.add(reply)
        choice = str(reply.data.get("choice", "")).strip()
        valid = {str(o.option_id) for o in sample.options}
        if choice not in valid:
            rescued = {v.upper(): v for v in valid}.get(choice.upper())
            choice = rescued or choice
        if choice in valid:
            result = {"status": "success", "prediction": choice}
        else:
            result = {
                "status": "failed",
                "prediction": "",
                "error_type": "InvalidChoice",
                "error_message": f"model chose '{choice}', not a valid option id",
            }
    except Exception as exc:
        result = {
            "status": "failed",
            "prediction": "",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    result["stats"] = stats.as_dict()
    output["result"] = result
    return output
