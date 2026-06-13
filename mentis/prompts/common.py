from __future__ import annotations

import json
from typing import Any


def prompt_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def prompt_header(task: str) -> str:
    return (
        f"Task: {task}\n"
        "Return only valid JSON. Do not use markdown. Keep reasoning concise.\n"
    )
