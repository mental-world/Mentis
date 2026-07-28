from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from .config import Settings
from .media import (
    extract_audio_wav,
    frame_preamble,
    image_content_part,
    sample_video_frames,
    video_duration_seconds,
)
from .schema import normalize_payload
from .utils import parse_json_loose

RETRYABLE_MARKERS = (
    "429", "rate limit", "timeout", "timed out", "connection",
    "temporarily unavailable", "server error", "overloaded",
)


@dataclass
class LLMReply:
    data: dict[str, Any]
    raw: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    calls: int = 1


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._semaphore = asyncio.Semaphore(max(1, settings.max_concurrent_requests))
        self._client = None
        self._transcripts: dict[str, str] = {}

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(timeout=self.settings.request_timeout_seconds)
        return self._client

    async def json_call(
        self,
        *,
        prompt: str,
        schema: type | None = None,
        scene: dict[str, Any] | None = None,
    ) -> LLMReply:
        started = time.perf_counter()
        messages = [{"role": "user", "content": await self._build_content(prompt, scene or {})}]
        api_attempts = max(1, self.settings.max_retries + 1)
        schema_attempts = 1 + max(0, self.settings.schema_retries)
        usage: dict[str, Any] = {}
        calls = 0
        last_error: Exception | None = None
        attempt = 0
        while api_attempts > 0 and schema_attempts > 0:
            try:
                async with self._semaphore:
                    raw, call_usage = await self._chat(messages)
                calls += 1
                usage = _sum_usage(usage, call_usage)
            except Exception as exc:
                last_error = exc
                api_attempts -= 1
                if api_attempts > 0 and _retryable(exc):
                    await asyncio.sleep(self._delay(attempt))
                    attempt += 1
                    continue
                break
            try:
                data = self._parse(raw, schema)
                return LLMReply(
                    data=data,
                    raw=raw,
                    usage=usage,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    calls=calls,
                )
            except Exception as exc:
                last_error = exc
                schema_attempts -= 1
                if schema_attempts > 0:
                    await asyncio.sleep(self._delay(attempt))
                    attempt += 1
                    continue
                break
        raise last_error if last_error else RuntimeError("LLM call failed")

    async def _chat(self, messages: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "max_completion_tokens": self.settings.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.settings.temperature is not None:
            kwargs["temperature"] = self.settings.temperature
        response = await client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
        usage = {}
        if response.usage is not None:
            usage = response.usage.model_dump() if hasattr(response.usage, "model_dump") else dict(response.usage)
        return text, usage

    def _parse(self, raw: str, schema: type | None) -> dict[str, Any]:
        parsed = parse_json_loose(raw)
        if isinstance(parsed, list) and schema is not None:
            wrapper = {"ActionPlanBatch": "actions", "BranchScoreBatch": "scores"}.get(schema.__name__)
            if wrapper:
                parsed = {wrapper: parsed}
        if not isinstance(parsed, dict) or not parsed:
            raise ValueError("Model response did not contain a non-empty JSON object")
        if schema is None:
            return parsed
        parsed = normalize_payload(schema, parsed)
        return schema.model_validate(parsed).model_dump(mode="json")

    async def _build_content(self, prompt: str, scene: dict[str, Any]) -> Any:
        image_paths = scene.get("image_paths") or []
        video_path = scene.get("video_path") or ""
        if not image_paths and not video_path:
            return prompt
        parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in image_paths:
            parts.append(image_content_part(path))
        if video_path:
            frames, timestamps = await asyncio.to_thread(
                sample_video_frames, video_path, self.settings.video_max_frames
            )
            duration = video_duration_seconds(video_path)
            parts.append({"type": "text", "text": frame_preamble(timestamps, duration)})
            parts.extend(frames)
            transcript = await self._transcript(video_path)
            if transcript:
                parts.append(
                    {
                        "type": "text",
                        "text": (
                            "Video audio transcript (spoken dialogue and salient sounds, in order; "
                            "attribute each line to its speaker using the frames):\n" + transcript
                        ),
                    }
                )
        return parts

    async def _transcript(self, video_path: str) -> str:
        if not self.settings.transcribe_video_audio:
            return ""
        cached = self._transcripts.get(video_path)
        if cached is not None:
            return cached
        transcript = ""
        wav = await asyncio.to_thread(extract_audio_wav, video_path)
        if wav is not None:
            try:
                with open(wav, "rb") as fh:
                    response = await self._get_client().audio.transcriptions.create(
                        model=self.settings.transcription_model, file=fh
                    )
                transcript = str(getattr(response, "text", "") or "").strip()
            except Exception:
                transcript = ""
        self._transcripts[video_path] = transcript
        return transcript

    def _delay(self, attempt: int) -> float:
        delay = self.settings.retry_initial_delay_seconds * (2**attempt)
        return min(delay, self.settings.retry_max_delay_seconds)


def _retryable(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    if any(term in name for term in ("ratelimit", "timeout", "connection", "apierror")):
        return True
    text = str(exc).lower()
    return any(term in text for term in RETRYABLE_MARKERS)


def _sum_usage(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    total = dict(left)
    for key, value in (right or {}).items():
        if isinstance(value, (int, float)) and isinstance(total.get(key), (int, float)):
            total[key] = total[key] + value
        elif key not in total:
            total[key] = value
    return total
