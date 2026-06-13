"""Base LLM client interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    parsed: dict[str, Any]
    raw_text: str = ""
    model: str = ""
    latency_ms: float = 0.0
    token_usage: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    request_metadata: dict[str, Any] = field(default_factory=dict)


class BaseLLMClient:
    async def complete_json(
        self,
        *,
        task: str,
        prompt: str,
        model: str,
        context: dict[str, Any] | None = None,
        schema: type | None = None,
    ) -> LLMResponse:
        raise NotImplementedError
