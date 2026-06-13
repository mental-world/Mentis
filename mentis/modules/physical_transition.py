from __future__ import annotations

from mentis.prompts import build_physical_transition_prompt
from mentis.schema import CandidateAction, PhysicalState, WorldState
from mentis.utils.tracing import response_metadata

from .base import ModuleBase


class PhysicalTransitionModel(ModuleBase):
    async def predict(
        self, sample_id: str, world_state: WorldState, action: CandidateAction
    ) -> tuple[PhysicalState, dict]:
        physical_state = world_state.physical_state.as_dict()
        mental_state = world_state.mental_state.as_dict()
        physical_action = {
            "option_id": action.option_id,
            "physical_action_description": action.physical_action_description,
        }
        prompt = build_physical_transition_prompt(physical_state, mental_state, physical_action)
        response = await self.call_llm(
            sample_id=sample_id,
            task="physical_transition",
            prompt=prompt,
            model=self.config.models.world_model,
            context={
                "physical_state": physical_state,
                "mental_state": mental_state,
                "physical_action": physical_action,
            },
            schema=PhysicalState,
        )
        return PhysicalState.model_validate(response.parsed), response_metadata(response)
