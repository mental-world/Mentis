from __future__ import annotations

from mentis.prompts import build_mental_transition_prompt
from mentis.schema import CandidateAction, MentalState, WorldState
from mentis.utils.tracing import response_metadata

from .base import ModuleBase


class MentalTransitionModel(ModuleBase):
    async def predict(
        self,
        sample_id: str,
        world_state: WorldState,
        action: CandidateAction,
    ) -> tuple[MentalState, dict]:
        physical_state = world_state.physical_state.as_dict()
        mental_state = world_state.mental_state.as_dict()
        action_dict = action.as_dict()
        prompt = build_mental_transition_prompt(physical_state, mental_state, action_dict)
        response = await self.call_llm(
            sample_id=sample_id,
            task="mental_transition",
            prompt=prompt,
            model=self.config.models.world_model,
            context={
                "physical_state": physical_state,
                "mental_state": mental_state,
                "action": action_dict,
            },
            schema=MentalState,
        )
        return MentalState.model_validate(response.parsed), response_metadata(response)
