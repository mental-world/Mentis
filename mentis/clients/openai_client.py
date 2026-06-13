"""OpenAI Responses API client."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, get_args, get_origin

from mentis.config import MentisConfig
from mentis.schema import (
    empty_for_schema,
    is_template_schema,
    normalize_for_schema,
    validate_raw_template_json,
)
from mentis.utils.json_utils import extract_json_object, repair_json_output
from mentis.utils.media_utils import scene_to_openai_content

from .base import BaseLLMClient, LLMResponse


class OpenAIResponsesClient(BaseLLMClient):
    def __init__(self, config: MentisConfig) -> None:
        self.config = config
        self._semaphore = asyncio.Semaphore(max(1, int(config.api.max_concurrent_requests)))

    async def complete_json(
        self,
        *,
        task: str,
        prompt: str,
        model: str,
        context: dict[str, Any] | None = None,
        schema: type | None = None,
    ) -> LLMResponse:
        started = time.perf_counter()
        raw, usage, parsed, request_metadata = await self._complete_with_retry(
            model=model,
            prompt=prompt,
            context=context or {},
            schema=schema,
        )
        return LLMResponse(
            parsed=parsed,
            raw_text=raw,
            model=model,
            latency_ms=(time.perf_counter() - started) * 1000,
            token_usage=usage,
            request_metadata=request_metadata,
        )

    async def _complete_with_retry(
        self,
        *,
        model: str,
        prompt: str,
        context: dict[str, Any],
        schema: type | None,
    ) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
        max_attempts = self._max_attempts_for_schema(schema)
        last_raw = ""
        last_usage: dict[str, Any] = {}
        last_json: dict[str, Any] | None = None
        last_request_metadata: dict[str, Any] = {}
        last_error: Exception | None = None
        received_response = False

        for attempt in range(max_attempts):
            try:
                raw, usage, request_metadata = await self._call_with_limit(model, prompt, context)
                received_response = True
                last_request_metadata = request_metadata
                last_raw, last_usage, last_json = raw, usage, None
                parsed = self._extract_non_empty_json(raw)
                last_raw, last_usage, last_json = raw, usage, parsed
                self._validate_raw_json(parsed, schema)
                return raw, usage, self._fill_and_validate(parsed, schema), request_metadata
            except Exception as exc:
                last_error = exc
                if attempt < max_attempts - 1 and self._should_retry_exception(exc):
                    await asyncio.sleep(self._retry_delay(attempt))
                    continue
                if attempt < max_attempts - 1 and received_response:
                    await asyncio.sleep(self._retry_delay(attempt))
                    continue
                break

        if received_response and schema is not None:
            return (
                last_raw,
                last_usage,
                self._fallback_for_schema(last_json or {}, schema),
                last_request_metadata,
            )
        if last_error:
            raise last_error
        raise RuntimeError("OpenAI call failed without an exception")

    def _responses_call(
        self, model: str, prompt: str, context: dict[str, Any]
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        from openai import OpenAI  # type: ignore

        client = OpenAI(**self._client_kwargs())
        input_payload, request_metadata = self._input_payload(prompt, context)
        kwargs: dict[str, Any] = {
            "model": model,
            "input": input_payload,
            "max_output_tokens": self.config.generation.max_output_tokens,
        }
        if self.config.generation.temperature is not None:
            kwargs["temperature"] = self.config.generation.temperature
        if self.config.generation.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.config.generation.reasoning_effort}
        if self.config.generation.text_verbosity:
            kwargs["text"] = {"verbosity": self.config.generation.text_verbosity}
        response = client.responses.create(**kwargs)
        usage = {}
        if hasattr(response, "usage") and response.usage is not None:
            usage = response.usage.model_dump() if hasattr(response.usage, "model_dump") else dict(response.usage)
        if hasattr(response, "output_text"):
            return response.output_text, usage, request_metadata
        return json.dumps(response.model_dump(), ensure_ascii=False), usage, request_metadata

    async def _call_with_limit(
        self, model: str, prompt: str, context: dict[str, Any]
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        async with self._semaphore:
            return await asyncio.to_thread(self._responses_call, model, prompt, context)

    def _client_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"timeout": self.config.api.timeout_seconds}
        base_url = self.config.api.base_url
        if base_url:
            kwargs["base_url"] = base_url
        return kwargs

    def _input_payload(self, prompt: str, context: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        scene = context.get("media_scene")
        if not isinstance(scene, dict):
            scene = {}
        if scene.get("image_paths") or scene.get("video_path"):
            content = scene_to_openai_content(
                scene, prompt, max_video_frames=self.config.raw.get("video", {}).get("max_frames", 8)
            )
            payload = [{"role": "user", "content": content}]
            return payload, _payload_summary(payload, scene)
        return prompt, {"payload_type": "text", "input_text_count": 1, "input_image_count": 0}

    def _extract_non_empty_json(self, raw: str) -> dict[str, Any]:
        parsed = repair_json_output(raw)
        if parsed is None:
            parsed = extract_json_object(raw)
        if not isinstance(parsed, dict) or not parsed:
            raise ValueError("OpenAI response did not contain a non-empty JSON object")
        return parsed

    def _validate_raw_json(self, parsed: dict[str, Any], schema: type | None) -> None:
        if schema is not None:
            validate_raw_template_json(schema, parsed)
            schema.model_validate(parsed)

    def _fill_and_validate(self, parsed: dict[str, Any], schema: type | None) -> dict[str, Any]:
        if schema is not None:
            parsed = normalize_for_schema(schema, parsed)
            return schema.model_validate(parsed).model_dump(mode="json")
        return parsed

    def _fallback_for_schema(self, parsed: dict[str, Any], schema: type) -> dict[str, Any]:
        if is_template_schema(schema):
            fallback = parsed if parsed else empty_for_schema(schema)
            return self._fill_and_validate(fallback, schema)
        return self._fill_and_validate(_fill_pydantic_defaults(schema, parsed), schema)

    def _max_attempts_for_schema(self, schema: type | None) -> int:
        api_attempts = max(1, int(self.config.api.max_retries) + 1)
        if schema is None:
            return api_attempts
        retry_count = (
            self.config.api.template_schema_max_retries
            if is_template_schema(schema)
            else self.config.api.pydantic_schema_max_retries
        )
        schema_attempts = max(1, int(retry_count) + 1)
        return max(api_attempts, schema_attempts)

    def _retry_delay(self, attempt: int) -> float:
        delay = self.config.api.retry_initial_delay_seconds * (2**attempt)
        return min(delay, self.config.api.retry_max_delay_seconds)

    def _should_retry_exception(self, exc: Exception) -> bool:
        name = exc.__class__.__name__.lower()
        if any(term in name for term in ["ratelimit", "timeout", "connection", "apierror"]):
            return True
        text = str(exc).lower()
        return any(
            term in text
            for term in [
                "429",
                "rate limit",
                "concurrency limit",
                "timeout",
                "temporarily unavailable",
                "connection",
            ]
        )


def create_llm_client(config: MentisConfig) -> BaseLLMClient:
    if config.api.provider != "openai":
        raise ValueError("Only api.provider='openai' is supported")
    return OpenAIResponsesClient(config)


def _payload_summary(payload: Any, scene: dict[str, Any]) -> dict[str, Any]:
    content = []
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict) and isinstance(first.get("content"), list):
            content = first["content"]
    input_text_count = sum(
        1 for item in content if isinstance(item, dict) and item.get("type") == "input_text"
    )
    input_image_count = sum(
        1 for item in content if isinstance(item, dict) and item.get("type") == "input_image"
    )
    source_image_count = len(scene.get("image_paths") or [])
    video_path = scene.get("video_path") or ""
    return {
        "payload_type": "multimodal",
        "content_item_count": len(content),
        "input_text_count": input_text_count,
        "input_image_count": input_image_count,
        "source_image_count": source_image_count,
        "video_path_present": bool(video_path),
        "video_frame_count_sent": input_image_count - source_image_count if video_path else 0,
    }


def _fill_pydantic_defaults(schema: type, parsed: dict[str, Any]) -> dict[str, Any]:
    data = dict(parsed) if isinstance(parsed, dict) else {}
    fields = getattr(schema, "model_fields", {})
    for name, field in fields.items():
        keys = _field_input_keys(name, field)
        if any(key in data and data[key] is not None for key in keys):
            continue
        fallback = _field_default(field)
        key = str(getattr(field, "alias", None) or name)
        data[key] = fallback
    return data


def _field_input_keys(name: str, field: Any) -> list[str]:
    keys = [name]
    alias = getattr(field, "alias", None)
    if alias:
        keys.append(str(alias))
    validation_alias = getattr(field, "validation_alias", None)
    if isinstance(validation_alias, str):
        keys.append(validation_alias)
    choices = getattr(validation_alias, "choices", None)
    if choices:
        keys.extend(str(choice) for choice in choices)
    return list(dict.fromkeys(keys))


def _field_default(field: Any) -> Any:
    if not _is_pydantic_undefined(getattr(field, "default", None)):
        return field.default
    factory = getattr(field, "default_factory", None)
    if factory is not None:
        return factory()
    return _empty_value_for_annotation(getattr(field, "annotation", Any))


def _is_pydantic_undefined(value: Any) -> bool:
    return value.__class__.__name__ in {"PydanticUndefinedType", "_PydanticUndefinedType"}


def _empty_value_for_annotation(annotation: Any) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is list:
        return []
    if origin is dict:
        return {}
    if origin is None and hasattr(annotation, "model_validate"):
        return {}
    if origin is not None and type(None) in args:
        non_none = [arg for arg in args if arg is not type(None)]
        return _empty_value_for_annotation(non_none[0]) if non_none else None
    if annotation is str:
        return ""
    if annotation in {int, float}:
        return 0.0 if annotation is float else 0
    if annotation is bool:
        return False
    return None
