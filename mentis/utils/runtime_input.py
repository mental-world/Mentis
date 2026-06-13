"""Runtime input normalization helpers for Mentis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from mentis.schema import DirectAnswerInput, OptionInput, SampleInput


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    if "sample_id" in normalized:
        normalized["sample_id"] = str(normalized["sample_id"])
    return normalized


def runtime_sample_from_input(sample: SampleInput, input_base_dir: str = "") -> dict[str, Any]:
    data = sample.as_dict()
    data["sample_id"] = str(data["sample_id"])
    data["task_type"] = sample.scene or sample.domain
    data["target_agent_description"] = data.get("target_agent") or ""
    media_scene = scene_from_story(data.get("story"))
    data["_media_scene"] = (
        resolve_scene_media_paths(media_scene, input_base_dir)
        if input_base_dir
        else media_scene
    )
    return data


def scene_from_story(story: Any) -> dict[str, Any]:
    if not isinstance(story, dict):
        return {"text": "", "image_paths": [], "video_path": "", "modality": ""}
    return normalize_scene_dict(
        {
            "text": story.get("text") or "",
            "image_paths": image_paths_from_story(story),
            "video_path": str(story.get("video") or ""),
            "modality": modality_from_story(story),
        }
    )


def normalize_scene_dict(scene: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(scene)
    normalized["text"] = str(normalized.get("text") or "")
    normalized["image_paths"] = image_paths_from_story(normalized)
    normalized["video_path"] = str(normalized.get("video_path") or "")
    normalized["modality"] = normalized.get("modality") or infer_modality(normalized)
    return normalized


def image_paths_from_story(story: dict[str, Any]) -> list[str]:
    raw = story.get("images") or story.get("image_paths") or []
    if isinstance(raw, list):
        return [str(path) for path in raw if path]
    return []


def modality_from_story(story: dict[str, Any]) -> str:
    if image_paths_from_story(story):
        return "image"
    if story.get("video"):
        return "video"
    return "text" if story.get("text") else ""


def infer_modality(scene: dict[str, Any]) -> str:
    if scene.get("image_paths"):
        return "image"
    if scene.get("video_path"):
        return "video"
    if scene.get("text"):
        return "text"
    return ""


def resolve_scene_media_paths(scene: dict[str, Any], base_dir: str) -> dict[str, Any]:
    resolved = dict(scene)
    base = Path(base_dir)
    resolved["image_paths"] = [
        resolve_media_path(path, base) for path in scene.get("image_paths", [])
    ]
    if scene.get("video_path"):
        resolved["video_path"] = resolve_media_path(str(scene["video_path"]), base)
    return resolved


def resolve_media_path(path: str, base: Path) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    candidate = base / p
    if candidate.exists():
        return str(candidate)
    cwd_candidate = Path.cwd() / p
    return str(cwd_candidate) if cwd_candidate.exists() else path


def direct_answer_input_from_sample(sample: dict[str, Any]) -> DirectAnswerInput:
    modality = direct_answer_modality(sample)
    return DirectAnswerInput(
        modality=modality,
        scene=direct_answer_text_scene(sample) if modality == "text" else {},
        question=sample.get("question", ""),
        options=[OptionInput.model_validate(opt) for opt in sample.get("options", [])],
    )


def direct_answer_modality(
    sample: dict[str, Any],
) -> Literal["text", "image", "video", "unknown"]:
    media_scene = sample.get("_media_scene") if isinstance(sample.get("_media_scene"), dict) else {}
    modality = str(media_scene.get("modality") or "").strip()
    if modality in {"text", "image", "video"}:
        return modality  # type: ignore[return-value]
    story = sample.get("story") if isinstance(sample.get("story"), dict) else {}
    if image_paths_from_story(story):
        return "image"
    if story.get("video"):
        return "video"
    if story.get("text"):
        return "text"
    return "unknown"


def direct_answer_text_scene(sample: dict[str, Any]) -> dict[str, Any]:
    story = sample.get("story") if isinstance(sample.get("story"), dict) else {}
    return {"text": story.get("text") or ""}
