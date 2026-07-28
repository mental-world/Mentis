from __future__ import annotations

from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LooseModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class Story(LooseModel):
    model_config = ConfigDict(extra="ignore")

    text: str = ""
    images: list[str] = Field(default_factory=list)
    video: Optional[str] = None
    scene_context: str = ""

    @field_validator("text", "scene_context", mode="before")
    @classmethod
    def _none_to_text(cls, value: Any) -> Any:
        return "" if value is None else value

    @field_validator("images", mode="before")
    @classmethod
    def _none_to_list(cls, value: Any) -> Any:
        return [] if value is None else value


class Option(LooseModel):
    model_config = ConfigDict(extra="ignore")

    option_id: str
    action_description: str


class Sample(LooseModel):
    model_config = ConfigDict(extra="ignore")

    sample_id: str
    modality: str = ""
    domain: str = ""
    domain_category: str = ""
    scene: str = ""
    scene_category: str = ""
    num_of_characters: Union[str, int] = ""
    target_agent: str = ""
    question: str = ""
    options: list[Option]
    story: Story
    answer: str = ""

    @field_validator("sample_id", "answer", mode="before")
    @classmethod
    def _to_text(cls, value: Any) -> Any:
        return "" if value is None else str(value)

    def option_ids(self) -> list[str]:
        return [str(option.option_id) for option in self.options]


class PhysicalState(LooseModel):
    entity_and_attribute: Optional[dict[str, Any]] = None
    relations: Optional[dict[str, Any]] = None
    environment: Optional[dict[str, Any]] = None


class MentalState(LooseModel):
    mental_entity_and_attribute: Optional[dict[str, Any]] = None
    relations: Optional[dict[str, Any]] = None
    atmosphere_of_environment: Optional[str] = None


class WorldState(LooseModel):
    model_config = ConfigDict(extra="ignore")

    physical_state: PhysicalState = Field(default_factory=PhysicalState)
    mental_state: MentalState = Field(default_factory=MentalState)

    def as_dict(self) -> dict[str, Any]:
        return {
            "physical_state": self.physical_state.as_dict(),
            "mental_state": self.mental_state.as_dict(),
        }


class Observation(LooseModel):
    model_config = ConfigDict(extra="ignore")

    physical_observation: PhysicalState = Field(default_factory=PhysicalState)
    mental_observation: MentalState = Field(default_factory=MentalState)

    @field_validator("physical_observation", "mental_observation", mode="before")
    @classmethod
    def _none_to_empty(cls, value: Any) -> Any:
        return {} if value is None else value


class ActionPlan(LooseModel):
    model_config = ConfigDict(extra="ignore")

    option_id: str
    description: str = ""
    physical_action: str = ""
    mental_action: str = ""

    @field_validator("physical_action", "mental_action", mode="before")
    @classmethod
    def _blank_absent(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str) and value.strip().lower() in {"none", "null", "n/a"}:
            return ""
        return value


class ActionPlanBatch(LooseModel):
    model_config = ConfigDict(extra="ignore")

    actions: list[ActionPlan] = Field(default_factory=list)


class BranchScore(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    option_id: str = ""
    mentally_consistent: float = 0.0
    physically_plausible: float = 0.0
    socially_appropriate: float = 0.0
    safety_veto: bool = False
    relative_rank: Optional[int] = None
    reasoning: Union[dict[str, Any], str] = ""
    weighted_score: float = 0.0

    @field_validator("mentally_consistent", "physically_plausible", "socially_appropriate", mode="before")
    @classmethod
    def _to_unit_interval(cls, value: Any) -> float:
        if isinstance(value, str):
            value = float(value.strip())
        value = float(value)
        if 1.0 < value <= 5.0:
            value = round((value - 1.0) / 4.0, 4)
        return min(1.0, max(0.0, value))

    def as_dict(self) -> dict[str, Any]:
        reasoning = self.reasoning
        if isinstance(reasoning, dict):
            reasoning = " ".join(str(part) for part in reasoning.values() if part)
        return {
            "mentally_consistent": self.mentally_consistent,
            "physically_plausible": self.physically_plausible,
            "socially_appropriate": self.socially_appropriate,
            "safety_veto": self.safety_veto,
            "relative_rank": self.relative_rank,
            "weighted_score": self.weighted_score,
            "reasoning": reasoning,
        }


class BranchScoreBatch(LooseModel):
    model_config = ConfigDict(extra="ignore")

    scores: list[BranchScore] = Field(default_factory=list)


class DirectChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    choice: str


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

MENTAL_STATE_TEMPLATE: dict[str, Any] = {
    "mental_entity_and_attribute": {
        "individual": [
            {
                "identity_attributes": {"name": None, "occupation": None},
                "appearance": None,
                "epistemic_state": {"beliefs": [None], "attention_focus": None},
                "motivational_state": {"goals": [None], "intentions": [None]},
                "affective_state": {"emotions": [None]},
                "dispositional_state": {"preferences": [None]},
                "normative_state": {"rules": [None], "customs": [None]},
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

WORLD_STATE_TEMPLATE: dict[str, Any] = {
    "physical_state": PHYSICAL_STATE_TEMPLATE,
    "mental_state": MENTAL_STATE_TEMPLATE,
}

OBSERVATION_TEMPLATE: dict[str, Any] = {
    "physical_observation": PHYSICAL_STATE_TEMPLATE,
    "mental_observation": MENTAL_STATE_TEMPLATE,
}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _unwrap(value: dict[str, Any], wrapper_key: str) -> dict[str, Any]:
    inner = value.get(wrapper_key)
    if isinstance(inner, dict) and len(value) == 1:
        return _unwrap(inner, wrapper_key)
    return value


def normalize_world_state(payload: Any) -> dict[str, Any]:
    data = _as_dict(payload)
    return {
        "physical_state": _unwrap(_as_dict(data.get("physical_state")), "physical_state"),
        "mental_state": _unwrap(_as_dict(data.get("mental_state")), "mental_state"),
    }


def normalize_physical_state(payload: Any) -> dict[str, Any]:
    return _unwrap(_as_dict(payload), "physical_state")


def normalize_mental_state(payload: Any) -> dict[str, Any]:
    return _unwrap(_as_dict(payload), "mental_state")


def normalize_observation(payload: Any) -> dict[str, Any]:
    data = {str(key).strip(): value for key, value in _as_dict(payload).items()}
    return {
        "physical_observation": _unwrap(_as_dict(data.get("physical_observation")), "physical_observation"),
        "mental_observation": _unwrap(_as_dict(data.get("mental_observation")), "mental_observation"),
    }


NORMALIZERS = {
    "WorldState": normalize_world_state,
    "PhysicalState": normalize_physical_state,
    "MentalState": normalize_mental_state,
    "Observation": normalize_observation,
}


def normalize_payload(schema: type | None, payload: dict[str, Any]) -> dict[str, Any]:
    normalizer = NORMALIZERS.get(getattr(schema, "__name__", ""))
    return normalizer(payload) if normalizer else payload
