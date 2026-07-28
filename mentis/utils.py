from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_json(path: str | Path, data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


async def gather_limited(
    items: list[T], worker: Callable[[T], Awaitable[R]], limit: int
) -> list[R]:
    semaphore = asyncio.Semaphore(max(1, limit))

    async def bounded(item: T) -> R:
        async with semaphore:
            return await worker(item)

    return await asyncio.gather(*(bounded(item) for item in items))


def extract_json_block(raw: str) -> str | None:
    starts = [i for i in (raw.find("{"), raw.find("[")) if i >= 0]
    if not starts:
        return None
    start = min(starts)
    opening = raw[start]
    closing = "}" if opening == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(raw)):
        char = raw[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return raw[start : index + 1]
    end = raw.rfind(closing)
    return raw[start : end + 1] if end > start else None


def parse_json_loose(raw: str) -> Any:
    try:
        return json.loads(raw)
    except Exception:
        pass
    block = extract_json_block(raw)
    if not block:
        return None
    cleaned = block.replace("```json", "").replace("```", "")
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
    try:
        return json.loads(cleaned)
    except Exception:
        return None
