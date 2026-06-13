from __future__ import annotations

from mentis.prompts import build_state_parser_prompt
from mentis.schema import WorldState
from mentis.utils.tracing import response_metadata

from .base import ModuleBase


class StateParser(ModuleBase):
    async def parse(
        self,
        *,
        sample_id: str,
        state_input: dict,
        media_scene: dict | None = None,
    ) -> tuple[WorldState, dict]:
        prompt = build_state_parser_prompt(state_input)
        response = await self.call_llm(
            sample_id=sample_id,
            task="state_parser",
            prompt=prompt,
            model=self.config.models.parser_model,
            context={"media_scene": media_scene or {}},
            schema=WorldState,
        )
        return WorldState.model_validate(response.parsed), response_metadata(response)
