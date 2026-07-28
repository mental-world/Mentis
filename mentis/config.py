from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional


def default_score_weights() -> dict[str, float]:
    return {
        "mentally_consistent": 0.45,
        "physically_plausible": 0.35,
        "socially_appropriate": 0.20,
    }


@dataclass
class Settings:
    model: str = "gpt-5.5"
    transcription_model: str = "whisper-1"
    temperature: Optional[float] = None
    max_output_tokens: int = 16384
    max_concurrent_requests: int = 4
    branch_concurrency: int = 6
    max_retries: int = 3
    schema_retries: int = 1
    request_timeout_seconds: float = 300.0
    retry_initial_delay_seconds: float = 2.0
    retry_max_delay_seconds: float = 30.0
    video_max_frames: int = 16
    transcribe_video_audio: bool = True
    score_weights: dict[str, float] = field(default_factory=default_score_weights)


def load_settings(path: str | Path | None = None) -> Settings:
    if path is None:
        return Settings()
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    known = {f.name for f in fields(Settings)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(unknown)}")
    return Settings(**data)


def load_env_file(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def require_api_key() -> None:
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise SystemExit(
            "OPENAI_API_KEY is not set. Export it or put it in a .env file "
            "(see .env.example)."
        )
