from __future__ import annotations

from mentis.schema import OptionInput, SampledAction


class TargetPseudoAgent:
    def __init__(self, expected_action_count: int = 6) -> None:
        self.expected_action_count = expected_action_count

    def sample_actions(
        self,
        *,
        options: list[OptionInput],
    ) -> tuple[list[SampledAction], dict]:
        actions = [
            SampledAction(
                option_id=option.option_id,
                action_description=option.action_description,
            )
            for option in options
        ]
        warnings = []
        if len(actions) != self.expected_action_count:
            warnings.append(
                f"expected {self.expected_action_count} test actions, got {len(actions)}"
            )
        return actions, {
            "mode": "multiple_choice_proxy",
            "action_count": len(actions),
            "warnings": warnings,
        }
