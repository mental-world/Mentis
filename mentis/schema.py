"""Data contracts and canonical state templates for Mentis."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _dump(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json", by_alias=False)


def _reasoning_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        parts = [str(part) for part in value.values() if part]
        return " ".join(parts)
    return str(value) if value is not None else ""


class MentisBaseModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    def as_dict(self) -> dict[str, Any]:
        return _dump(self)


class StoryInput(MentisBaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    text: str = ""
    images: list[str] = Field(default_factory=list)
    video: Optional[str] = None


class OptionInput(MentisBaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    option_id: str
    action_description: str


class SampleInput(MentisBaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    sample_id: str
    domain: str = ""
    domain_category: str = ""
    scene: str = ""
    num_of_characters: Union[str, int] = ""
    story: StoryInput
    target_agent: str
    question: str
    options: list[OptionInput]
    difficulty: Optional[str] = None
    golden_answer: Optional[dict[str, Any]] = None


class PhysicalState(MentisBaseModel):
    entity_and_attribute: Optional[dict[str, Any]] = None
    relations: Optional[dict[str, Any]] = None
    environment: Optional[dict[str, Any]] = None


class MentalState(MentisBaseModel):
    mental_entity_and_attribute: Optional[dict[str, Any]] = None
    relations: Optional[dict[str, Any]] = None
    atmosphere_of_environment: Optional[str] = None


class WorldState(MentisBaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    physical_state: PhysicalState = Field(default_factory=PhysicalState)
    mental_state: MentalState = Field(default_factory=MentalState)

    def as_state_dict(self) -> dict[str, Any]:
        return {
            "physical_state": self.physical_state.as_dict(),
            "mental_state": self.mental_state.as_dict(),
        }


class TargetObservation(MentisBaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    physical_observation: PhysicalState = Field(default_factory=PhysicalState)
    mental_observation: MentalState = Field(default_factory=MentalState)

    @field_validator("physical_observation", "mental_observation", mode="before")
    @classmethod
    def _empty_observation_for_none(cls, value: Any) -> Any:
        return {} if value is None else value

    def as_observation_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class SampledAction(MentisBaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    option_id: str
    action_description: str


class ActionDecomposition(MentisBaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    physical_action_description: str = ""
    mental_action_description: str = ""

    @field_validator("physical_action_description", "mental_action_description", mode="before")
    @classmethod
    def _empty_for_absent_action(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str) and value.strip().lower() in {"none", "null", "n/a"}:
            return ""
        return value


class CandidateAction(MentisBaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    option_id: str
    raw_action_description: str
    physical_action_description: str = ""
    mental_action_description: str = ""

    @field_validator(
        "physical_action_description",
        "mental_action_description",
        mode="before",
    )
    @classmethod
    def _empty_for_absent_action(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str) and value.strip().lower() in {"none", "null", "n/a"}:
            return ""
        return value


class ScoreResult(MentisBaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    option_id: str = ""
    mentally_consistent: float
    physically_plausible: float
    socially_appropriate: float
    safety_legality_veto: bool
    reasoning: Union[dict[str, Any], str] = ""
    raw_value_score: float = 0.0
    overall_score: float = 0.0

    @field_validator(
        "mentally_consistent",
        "physically_plausible",
        "socially_appropriate",
        mode="before",
    )
    @classmethod
    def _score_to_float(cls, value: Any) -> float:
        if isinstance(value, str):
            value = float(value.strip())
        value = float(value)
        if 1.0 < value <= 5.0:
            return round((value - 1.0) / 4.0, 4)
        return value

    def as_score_dict(self) -> dict[str, Any]:
        return {
            "mentally_consistent": self.mentally_consistent,
            "physically_plausible": self.physically_plausible,
            "socially_appropriate": self.socially_appropriate,
            "safety_legality_veto": self.safety_legality_veto,
            "raw_value_score": self.raw_value_score,
            "overall_score": self.overall_score,
            "reasoning": _reasoning_text(self.reasoning),
        }


class DirectAnswerResult(MentisBaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    final_action: str


class DirectAnswerInput(MentisBaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    modality: Literal["text", "image", "video", "unknown"] = "unknown"
    scene: dict[str, Any] = Field(default_factory=dict)
    question: str = ""
    options: list[OptionInput] = Field(default_factory=list)

    def option_ids(self) -> set[str]:
        return {str(option.option_id) for option in self.options}


class GeneratedResults(MentisBaseModel):
    status: Literal["success", "failed"] = "success"
    current_state_s_t: dict[str, Any] = Field(default_factory=lambda: WorldState().as_state_dict())
    target_agent_observation_o_t: dict[str, Any] = Field(
        default_factory=lambda: TargetObservation().as_observation_dict()
    )
    candidate_actions: list[dict[str, Any]] = Field(default_factory=list)
    next_state_s_t1: dict[str, Any] = Field(
        default_factory=dict,
        alias="next_state_s_{t+1}",
    )
    score: dict[str, Any] = Field(default_factory=dict)
    final_action: str = ""
    decision_trace: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    @classmethod
    def failed(cls, *, error_type: str, error_message: str, metadata: dict[str, Any]) -> "GeneratedResults":
        return cls(
            status="failed",
            current_state_s_t=WorldState().as_state_dict(),
            target_agent_observation_o_t=TargetObservation().as_observation_dict(),
            metadata=metadata,
            error_type=error_type,
            error_message=error_message,
        )

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)


class JudgeResult(MentisBaseModel):
    item: str = ""
    physical_score: float = 0.0
    mental_score: float = 0.0
    semantic_consistency_score: float = 0.0
    coupling_score: Optional[float] = None
    transition_score: Optional[float] = None
    error_types: list[str] = Field(default_factory=list)
    reasoning: str = ""
    option_scores: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0


PHYSICAL_STATE_TEMPLATE: dict[str, Any] = {
    "entity_and_attribute": {
        "objects": [
            {
                "intrinsic_property": {"appearance": None},
                "contextual_attributes": {
                    "position": None,
                    "physical_condition": None,
                    "semantic_content": None,
                },
            }
        ],
        "characters": [
            {
                "intrinsic_property": {"appearance": None},
                "contextual_attributes": {
                    "facial_expression": None,
                    "gaze": None,
                    "body_motion": None,
                    "position": None,
                    "semantic_content": None,
                },
            }
        ],
    },
    "relations": {
        "spatial_relations": {
            "relative_position": [None],
            "distance": [None],
            "occlusion": [None],
        },
        "contact_relations": [None],
    },
    "environment": {
        "place": None,
        "ambient_condition": {
            "acoustic_condition": {"background_sound": None},
            "space_condition": {"clutter": None},
        },
    },
}


PHYSICAL_TRANSITION_SCHEMA: dict[str, Any] = PHYSICAL_STATE_TEMPLATE


MENTAL_STATE_TEMPLATE: dict[str, Any] = {
    "mental_entity_and_attribute": {
        "individual": [
            {
                "identity_attributes": {
                    "name": None,
                    "occupation": None,
                },
                "appearance": None,
                "epistemic_state": {
                    "beliefs": [None],
                    "attention_focus": None,
                },
                "motivational_state": {
                    "goals": [None],
                    "intentions": [None],
                },
                "affective_state": {"emotions": [None]},
                "dispositional_state": {"preferences": [None]},
                "normative_state": {
                    "rules": [None],
                    "customs": [None],
                },
            }
        ]
    },
    "relations": {
        "attitudes": {
            "person_object_attitude": [None],
            "person_person_attitude": [None],
        },
        "role_relations": {"person_person_role_relation": [None]},
    },
    "atmosphere_of_environment": None,
}


MENTAL_TRANSITION_SCHEMA: dict[str, Any] = MENTAL_STATE_TEMPLATE


WORLD_STATE_TEMPLATE: dict[str, Any] = {
    "physical_state": PHYSICAL_STATE_TEMPLATE,
    "mental_state": MENTAL_STATE_TEMPLATE,
}


TARGET_OBSERVATION_TEMPLATE: dict[str, Any] = {
    "physical_observation": PHYSICAL_STATE_TEMPLATE,
    "mental_observation": MENTAL_STATE_TEMPLATE,
}


def is_template_schema(schema: type | None) -> bool:
    return getattr(schema, "__name__", "") in {
        "WorldState",
        "PhysicalState",
        "MentalState",
        "TargetObservation",
    }


def empty_for_schema(schema: type | None) -> dict[str, Any]:
    name = getattr(schema, "__name__", "")
    if name == "WorldState":
        return normalize_world_state({})
    if name == "PhysicalState":
        return {}
    if name == "MentalState":
        return {}
    if name == "TargetObservation":
        return normalize_target_observation({})
    return {}


def normalize_for_schema(schema: type | None, parsed: dict[str, Any]) -> dict[str, Any]:
    if schema is None:
        return parsed
    name = getattr(schema, "__name__", "")
    if name == "WorldState":
        return normalize_world_state(parsed)
    if name == "PhysicalState":
        return normalize_physical_state(parsed)
    if name == "MentalState":
        return normalize_mental_state(parsed)
    if name == "TargetObservation":
        return normalize_target_observation(parsed)
    return parsed


def validate_raw_template_json(schema: type | None, parsed: dict[str, Any]) -> None:
    name = getattr(schema, "__name__", "")
    if name == "WorldState":
        _validate_optional_dict(parsed, "physical_state", "WorldState")
        _validate_optional_dict(parsed, "mental_state", "WorldState")
    elif name == "TargetObservation":
        _validate_raw_target_observation(parsed)


def normalize_world_state(value: dict[str, Any]) -> dict[str, Any]:
    data = value if isinstance(value, dict) else {}
    return {
        "physical_state": _dict_or_empty(data.get("physical_state")),
        "mental_state": _dict_or_empty(data.get("mental_state")),
    }


def normalize_physical_state(value: dict[str, Any]) -> dict[str, Any]:
    return _dict_or_empty(value)


def normalize_mental_state(value: dict[str, Any]) -> dict[str, Any]:
    return _dict_or_empty(value)


def normalize_target_observation(value: dict[str, Any]) -> dict[str, Any]:
    data = _strip_dict_keys(value if isinstance(value, dict) else {})
    return {
        "physical_observation": PhysicalState.model_validate(
            _dict_or_empty(data.get("physical_observation"))
        ).as_dict(),
        "mental_observation": MentalState.model_validate(
            _dict_or_empty(data.get("mental_observation"))
        ).as_dict(),
    }


def fill_template(value: Any, template: Any) -> Any:
    if isinstance(template, dict):
        source = value if isinstance(value, dict) else {}
        return {
            key: fill_template(source[key], sub_template)
            if key in source
            else _missing_value(sub_template)
            for key, sub_template in template.items()
        }
    if isinstance(template, list):
        if not isinstance(value, list):
            return []
        if not template:
            return deepcopy(value)
        item_template = template[0]
        if item_template is None:
            return deepcopy(value)
        return [fill_template(item, item_template) for item in value]
    return deepcopy(value)


def _missing_value(template: Any) -> Any:
    if isinstance(template, dict):
        return fill_template({}, template)
    if isinstance(template, list):
        return []
    return None


def _strip_dict_keys(value: dict[str, Any]) -> dict[str, Any]:
    return {str(key).strip(): val for key, val in value.items()}


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _validate_optional_dict(value: dict[str, Any], key: str, label: str) -> None:
    if key in value and not isinstance(value[key], dict):
        raise ValueError(f"{label}.{key} must be an object when present")


def _validate_raw_target_observation(value: dict[str, Any]) -> None:
    data = _strip_dict_keys(value)
    _validate_optional_dict(data, "physical_observation", "TargetObservation")
    _validate_optional_dict(data, "mental_observation", "TargetObservation")
