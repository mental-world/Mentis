"""JSON IO and repair helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def save_json(data: Any, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if raw.strip():
            records.append(json.loads(raw))
    return records


def read_records(path: str | Path) -> tuple[list[dict[str, Any]], bool]:
    p = Path(path)
    if p.suffix.lower() != ".jsonl":
        raise ValueError(f"Input records must be JSONL, got: {p}")
    return _load_jsonl(p), True


def repair_json_output(raw: str) -> dict[str, Any] | list[Any] | None:
    try:
        return json.loads(raw)
    except Exception:
        pass
    candidate = extract_json_text(raw)
    if not candidate:
        return None
    cleaned = re.sub(r",(\s*[}\]])", r"\1", candidate)
    cleaned = cleaned.replace("```json", "").replace("```", "")
    try:
        return json.loads(cleaned)
    except Exception:
        return None


def extract_json_text(raw: str) -> str | None:
    start_obj = raw.find("{")
    start_arr = raw.find("[")
    starts = [x for x in [start_obj, start_arr] if x >= 0]
    if not starts:
        return None
    start = min(starts)
    closing = "}" if raw[start] == "{" else "]"
    end = raw.rfind(closing)
    if end <= start:
        return None
    return raw[start : end + 1]


def extract_json_object(raw: str) -> dict[str, Any] | None:
    repaired = repair_json_output(raw)
    return repaired if isinstance(repaired, dict) else None
