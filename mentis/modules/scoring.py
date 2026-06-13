from __future__ import annotations

from mentis.policies.ablation import AblationPolicy, get_ablation_policy
from mentis.prompts import build_scoring_prompt
from mentis.schema import CandidateAction, ScoreResult, TargetObservation, WorldState
from mentis.schema import normalize_target_observation
from mentis.utils.tracing import response_metadata

from .base import ModuleBase


class ScoringModule(ModuleBase):
    async def score(
        self,
        *,
        sample_id: str,
        world_state: WorldState,
        observation: TargetObservation,
        question: str,
        target_agent: str,
        action: CandidateAction,
        next_state: WorldState,
        ablation: str | AblationPolicy = "full_mwm",
    ) -> tuple[ScoreResult, dict]:
        policy = (
            ablation
            if isinstance(ablation, AblationPolicy)
            else get_ablation_policy(ablation)
        )
        world_state_for_eval = policy.state_for_evaluator(world_state.as_dict())
        observation_for_eval = policy.observation_for_evaluator(
            normalize_target_observation(observation.as_observation_dict())
        )
        next_state_for_eval = policy.state_for_evaluator(next_state.as_dict())
        prompt = build_scoring_prompt(
            target_agent,
            world_state_for_eval,
            observation_for_eval,
            question,
            action.as_dict(),
            next_state_for_eval,
        )
        response = await self.call_llm(
            sample_id=sample_id,
            task="scoring",
            prompt=prompt,
            model=self.config.models.scoring_model,
            context={
                "ablation": policy.name,
                "world_state": world_state_for_eval,
                "observation": observation_for_eval,
                "question": question,
                "target_agent": target_agent,
                "action": action.as_dict(),
                "next_state": next_state_for_eval,
            },
            schema=ScoreResult,
        )
        result = ScoreResult.model_validate(response.parsed)
        result.option_id = action.option_id
        result.raw_value_score = self.raw_value_score(result)
        result.overall_score = self.weighted_score(result)
        return result, response_metadata(response)

    def raw_value_score(self, score: ScoreResult) -> float:
        weights = self.config.scoring.weights
        return round(
            float(weights.get("mentally_consistent", 0.0)) * score.mentally_consistent
            + float(weights.get("physically_plausible", 0.0)) * score.physically_plausible
            + float(weights.get("socially_appropriate", 0.0)) * score.socially_appropriate,
            4,
        )

    def weighted_score(self, score: ScoreResult) -> float:
        if score.safety_legality_veto:
            return 0.0
        return score.raw_value_score or self.raw_value_score(score)
