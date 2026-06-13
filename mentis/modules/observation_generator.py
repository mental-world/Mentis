from __future__ import annotations

from mentis.prompts import build_observation_generation_prompt
from mentis.schema import TargetObservation, WorldState
from mentis.utils.tracing import response_metadata

from .base import ModuleBase


class ObservationGenerator(ModuleBase):
    async def generate(
        self,
        sample_id: str,
        world_state: WorldState,
        target_agent: str,
    ) -> tuple[TargetObservation, dict]:
        state = world_state.as_dict()
        prompt = build_observation_generation_prompt(state, target_agent)
        response = await self.call_llm(
            sample_id=sample_id,
            task="observation_generation",
            prompt=prompt,
            model=self.config.models.world_model,
            context={
                "world_state": state,
                "target_agent": target_agent,
            },
            schema=TargetObservation,
        )
        return TargetObservation.model_validate(response.parsed), response_metadata(response)
