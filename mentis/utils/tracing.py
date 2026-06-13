"""Run tracing and LLM call metadata helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4


class RunLogger:
    def __init__(self, output_dir: str, enabled: bool = True, run_id: str = "") -> None:
        self.run_id = run_id or uuid4().hex[:12]
        self.output_dir = Path(output_dir) / self.run_id
        self.enabled = enabled
        if enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def log_llm_call(
        self,
        *,
        sample_id: str,
        task: str,
        prompt: str,
        raw_output: str,
        parsed: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        if not self.enabled:
            return
        record = {
            "sample_id": sample_id,
            "task": task,
            "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt": prompt,
            "raw_output": raw_output,
            "parsed": parsed,
            "metadata": metadata,
        }
        path = self.output_dir / f"{sample_id}_llm_calls.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def response_metadata(response: Any, *, call_count: int = 1) -> dict[str, Any]:
    metadata = {
        "model": response.model,
        "latency_ms": response.latency_ms,
        "token_usage": response.token_usage,
        "warnings": response.warnings,
        "call_count": call_count,
    }
    request_metadata = getattr(response, "request_metadata", None)
    if request_metadata:
        metadata["request"] = request_metadata
    return metadata


def merge_metadata(items: list[dict[str, Any]]) -> dict[str, Any]:
    warnings: list[str] = []
    latency_ms = 0.0
    models: list[str] = []
    token_usage: dict[str, Any] = {}
    call_count = 0
    for item in items:
        warnings.extend(item.get("warnings", []))
        latency_ms += float(item.get("latency_ms", 0.0) or 0.0)
        call_count += int(item.get("call_count", 1 if item.get("model") else 0))
        if item.get("model"):
            models.append(str(item["model"]))
        token_usage = sum_token_usage([token_usage, item.get("token_usage", {})])
    return {
        "model": ",".join(sorted(set(models))),
        "latency_ms": latency_ms,
        "token_usage": token_usage,
        "warnings": sorted(set(warnings)),
        "call_count": call_count,
    }


def sum_token_usage(items: Any) -> dict[str, Any]:
    total: dict[str, Any] = {}
    for usage in items:
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            if isinstance(value, (int, float)):
                total[key] = total.get(key, 0) + value
            elif key not in total:
                total[key] = value
    return total
