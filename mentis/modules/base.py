"""Shared module helpers."""

from __future__ import annotations

from typing import Any

from mentis.clients.base import BaseLLMClient, LLMResponse
from mentis.config import MentisConfig
from mentis.utils.tracing import RunLogger


class ModuleBase:
    def __init__(
        self, client: BaseLLMClient, config: MentisConfig, logger: RunLogger | None = None
    ) -> None:
        self.client = client
        self.config = config
        self.logger = logger

    async def call_llm(
        self,
        *,
        sample_id: str,
        task: str,
        prompt: str,
        model: str,
        context: dict[str, Any],
        schema: type | None = None,
    ) -> LLMResponse:
        response = await self.client.complete_json(
            task=task, prompt=prompt, model=model, context=context, schema=schema
        )
        if self.logger:
            self.logger.log_llm_call(
                sample_id=sample_id,
                task=task,
                prompt=prompt,
                raw_output=response.raw_text,
                parsed=response.parsed,
                metadata={
                    "model": response.model,
                    "latency_ms": response.latency_ms,
                    "token_usage": response.token_usage,
                    "warnings": response.warnings,
                    "request": response.request_metadata,
                },
            )
        return response
