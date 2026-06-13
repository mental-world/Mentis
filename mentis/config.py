"""Configuration loading for Mentis.

The project intentionally keeps configuration dependency-light. If PyYAML is
not installed, the small parser below handles the default YAML subset used by
this repository.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return {}
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    if value.startswith("[") or value.startswith("{"):
        try:
            return ast.literal_eval(value)
        except Exception:
            return value
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        key_value = raw.strip()
        if ":" not in key_value:
            continue
        key, value = key_value.split(":", 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        parsed = _parse_scalar(value)
        parent[key.strip()] = parsed
        if isinstance(parsed, dict) and value.strip() == "":
            stack.append((indent, parsed))
    return root


def load_yaml(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        return loaded or {}
    except Exception:
        return _parse_simple_yaml(text)


@dataclass
class APIConfig:
    provider: str = "openai"
    base_url: str = ""
    timeout_seconds: int = 90
    max_retries: int = 3
    max_concurrent_requests: int = 1
    template_schema_max_retries: int = 1
    pydantic_schema_max_retries: int = 1
    retry_initial_delay_seconds: float = 0.5
    retry_max_delay_seconds: float = 30.0


@dataclass
class ModelConfig:
    parser_model: str = "gpt-5.5"
    world_model: str = "gpt-5.5"
    scoring_model: str = "gpt-5.5"
    direct_answer_model: str = ""
    judge_model: str = "gpt-5.5"

    def __post_init__(self) -> None:
        if not self.direct_answer_model:
            self.direct_answer_model = self.scoring_model


@dataclass
class GenerationConfig:
    temperature: float | None = None
    max_output_tokens: int = 4096
    reasoning_effort: str = "high"
    text_verbosity: str = ""


@dataclass
class TransitionConfig:
    max_concurrency: int = 6
    expected_action_count: int = 6


@dataclass
class ScoringConfig:
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "mentally_consistent": 0.45,
            "physically_plausible": 0.35,
            "socially_appropriate": 0.2,
        }
    )
    tie_break_order: list[str] = field(
        default_factory=lambda: [
            "physically_plausible",
            "mentally_consistent",
            "socially_appropriate",
            "option_order",
        ]
    )


@dataclass
class EvaluationConfig:
    llm_judge_enabled: bool = True
    judge_max_concurrency: int = 2


@dataclass
class LoggingConfig:
    save_raw_llm_outputs: bool = True
    save_prompts: bool = True
    output_dir: str = "outputs/logs"


@dataclass
class MentisConfig:
    api: APIConfig = field(default_factory=APIConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    transition: TransitionConfig = field(default_factory=TransitionConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    raw: dict[str, Any] = field(default_factory=dict)


def _merge_dataclass(cls: type, data: dict[str, Any]) -> Any:
    allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in data.items() if k in allowed})


def _api_config(raw_api: dict[str, Any]) -> APIConfig:
    data = dict(raw_api)
    if "OPENAI_BASE_URL" in os.environ:
        data["base_url"] = os.environ["OPENAI_BASE_URL"].strip()
    return _merge_dataclass(APIConfig, data)


def load_config(path: str | Path = "configs/default.yaml") -> MentisConfig:
    raw = load_yaml(path)
    return MentisConfig(
        api=_api_config(raw.get("api", {})),
        models=_merge_dataclass(ModelConfig, raw.get("models", {})),
        generation=_merge_dataclass(GenerationConfig, raw.get("generation", {})),
        transition=_merge_dataclass(TransitionConfig, raw.get("transition", {})),
        scoring=_merge_dataclass(ScoringConfig, raw.get("scoring", {})),
        evaluation=_merge_dataclass(EvaluationConfig, raw.get("evaluation", {})),
        logging=_merge_dataclass(LoggingConfig, raw.get("logging", {})),
        raw=raw,
    )
