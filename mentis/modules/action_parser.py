from __future__ import annotations

from mentis.prompts import build_action_parser_prompt
from mentis.schema import (
    ActionDecomposition,
    CandidateAction,
    SampledAction,
)
from mentis.utils.concurrency_utils import gather_bounded
from mentis.utils.tracing import merge_metadata, response_metadata

from .base import ModuleBase


class ActionParser(ModuleBase):
    async def parse(
        self,
        *,
        sample_id: str,
        sampled_actions: list[SampledAction],
    ) -> tuple[list[CandidateAction], dict]:
        async def worker(action: SampledAction) -> tuple[CandidateAction, dict]:
            return await self._parse_one(sample_id, action)

        results = await gather_bounded(sampled_actions, worker, max(1, len(sampled_actions)))
        actions = [action for action, _ in results]
        return actions, merge_metadata([meta for _, meta in results])

    async def _parse_one(
        self, sample_id: str, sampled_action: SampledAction
    ) -> tuple[CandidateAction, dict]:
        prompt = build_action_parser_prompt(sampled_action.action_description)
        response = await self.call_llm(
            sample_id=sample_id,
            task="action_parser",
            prompt=prompt,
            model=self.config.models.parser_model,
            context={"action_description": sampled_action.action_description},
            schema=ActionDecomposition,
        )
        decomposition = ActionDecomposition.model_validate(response.parsed)
        action = CandidateAction(
            option_id=sampled_action.option_id,
            raw_action_description=sampled_action.action_description,
            physical_action_description=decomposition.physical_action_description,
            mental_action_description=decomposition.mental_action_description,
        )
        return action, response_metadata(response)
