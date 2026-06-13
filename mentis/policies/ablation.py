"""Centralized ablation behavior for Mentis runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AblationPolicy:
    name: str
    direct_answer: bool = False
    use_oracle_state: bool = False
    use_oracle_observation: bool = False
    hide_mental_from_evaluator: bool = False
    hide_physical_from_evaluator: bool = False

    def state_for_evaluator(self, state: dict[str, Any]) -> dict[str, Any]:
        if self.hide_mental_from_evaluator:
            return {key: value for key, value in state.items() if key != "mental_state"}
        if self.hide_physical_from_evaluator:
            return {key: value for key, value in state.items() if key != "physical_state"}
        return state

    def observation_for_evaluator(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self.hide_mental_from_evaluator:
            return {
                key: value
                for key, value in observation.items()
                if key != "mental_observation"
            }
        if self.hide_physical_from_evaluator:
            return {
                key: value
                for key, value in observation.items()
                if key != "physical_observation"
            }
        return observation


ABLATION_POLICIES: dict[str, AblationPolicy] = {
    "full_mwm": AblationPolicy("full_mwm"),
    "direct_answer_baseline": AblationPolicy(
        "direct_answer_baseline",
        direct_answer=True,
    ),
    "no_mental_information": AblationPolicy(
        "no_mental_information",
        hide_mental_from_evaluator=True,
    ),
    "no_physical_information": AblationPolicy(
        "no_physical_information",
        hide_physical_from_evaluator=True,
    ),
    "oracle_state": AblationPolicy(
        "oracle_state",
        use_oracle_state=True,
    ),
    "oracle_observation": AblationPolicy(
        "oracle_observation",
        use_oracle_observation=True,
    ),
}

ABLATION_CHOICES = tuple(ABLATION_POLICIES)


def get_ablation_policy(name: str) -> AblationPolicy:
    try:
        return ABLATION_POLICIES[name]
    except KeyError as exc:
        choices = ", ".join(ABLATION_CHOICES)
        raise ValueError(f"Unknown ablation '{name}'. Expected one of: {choices}") from exc
