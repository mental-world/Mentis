from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from . import prompts
from .config import Settings
from .llm import LLMClient, LLMReply
from .media import scene_from_story
from .schema import (
    ActionPlan,
    ActionPlanBatch,
    BranchScore,
    BranchScoreBatch,
    MentalState,
    Observation,
    PhysicalState,
    Sample,
    WorldState,
    normalize_observation,
)
from .utils import gather_limited


@dataclass
class Branch:
    plan: ActionPlan
    physical: PhysicalState | None = None
    mental: MentalState | None = None
    successor: WorldState | None = None
    score: BranchScore | None = None
    error: str = ""

    @property
    def failed(self) -> bool:
        return bool(self.error)


@dataclass
class RunStats:
    llm_calls: int = 0
    latency_ms: float = 0.0
    tokens: dict[str, Any] = field(default_factory=dict)

    def add(self, reply: LLMReply) -> None:
        self.llm_calls += reply.calls
        self.latency_ms += reply.latency_ms
        for key, value in reply.usage.items():
            if isinstance(value, (int, float)) and isinstance(self.tokens.get(key), (int, float)):
                self.tokens[key] += value
            elif key not in self.tokens:
                self.tokens[key] = value

    def as_dict(self) -> dict[str, Any]:
        return {
            "llm_calls": self.llm_calls,
            "latency_ms": round(self.latency_ms, 1),
            "tokens": self.tokens,
        }


class MentisEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.llm = LLMClient(settings)

    async def run_sample(self, record: dict[str, Any], base_dir: str = "") -> dict[str, Any]:
        output = public_record(record)
        stats = RunStats()
        try:
            sample = Sample.model_validate(record)
            scene = scene_from_story(sample.story.as_dict(), base_dir)
            result = await self._pipeline(sample, scene, stats)
        except Exception as exc:
            result = {
                "status": "failed",
                "prediction": "",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        result["stats"] = stats.as_dict()
        output["result"] = result
        return output

    async def _pipeline(self, sample: Sample, scene: dict[str, Any], stats: RunStats) -> dict[str, Any]:
        state = await self._parse_state(scene, stats)
        observation = await self._render_observation(state, sample.target_agent, stats)
        plans = await self._decompose_actions(sample, stats)
        branches = await gather_limited(
            plans,
            lambda plan: self._simulate_branch(state, plan, stats),
            min(len(plans), self.settings.branch_concurrency),
        )
        await self._score_branches(sample, state, observation, branches, stats)
        scored = [branch.score for branch in branches if branch.score is not None]
        if not scored:
            errors = "; ".join(f"{b.plan.option_id}: {b.error}" for b in branches if b.failed)
            return {
                "status": "failed",
                "prediction": "",
                "current_state": state.as_dict(),
                "target_observation": normalize_observation(observation.as_dict()),
                "candidate_actions": [plan.as_dict() for plan in plans],
                "error_type": "BranchExecutionError",
                "error_message": errors or "no branch survived to scoring",
            }
        prediction, trace = select_action(scored, self.settings.score_weights, sample.sample_id)
        failed_options = sorted(b.plan.option_id for b in branches if b.failed)
        if failed_options:
            trace["failed_branch_options"] = failed_options
        return {
            "status": "success",
            "prediction": prediction,
            "current_state": state.as_dict(),
            "target_observation": normalize_observation(observation.as_dict()),
            "candidate_actions": [plan.as_dict() for plan in plans],
            "successor_states": {
                b.plan.option_id: b.successor.as_dict() for b in branches if b.successor is not None
            },
            "score_table": {s.option_id: s.as_dict() for s in scored},
            "decision_trace": trace,
        }

    async def _parse_state(self, scene: dict[str, Any], stats: RunStats) -> WorldState:
        story_block: dict[str, Any] = {"modality": scene["modality"]}
        if scene["modality"] == "text":
            story_block["text"] = scene["text"]
        else:
            story_block["media"] = "(attached)"
            if scene.get("scene_context"):
                story_block["scene_context"] = scene["scene_context"]
        reply = await self.llm.json_call(
            prompt=prompts.state_prompt(story_block, scene["modality"]),
            schema=WorldState,
            scene=scene,
        )
        stats.add(reply)
        return WorldState.model_validate(reply.data)

    async def _render_observation(
        self, state: WorldState, target_agent: str, stats: RunStats
    ) -> Observation:
        reply = await self.llm.json_call(
            prompt=prompts.observation_prompt(state.as_dict(), target_agent),
            schema=Observation,
        )
        stats.add(reply)
        return Observation.model_validate(reply.data)

    async def _decompose_actions(self, sample: Sample, stats: RunStats) -> list[ActionPlan]:
        options = [
            {"option_id": option.option_id, "action_description": option.action_description}
            for option in sample.options
        ]
        try:
            reply = await self.llm.json_call(
                prompt=prompts.actions_prompt(options), schema=ActionPlanBatch
            )
            stats.add(reply)
            batch = ActionPlanBatch.model_validate(reply.data)
            by_id = {str(item.option_id): item for item in batch.actions}
            if set(by_id) != {str(o["option_id"]) for o in options}:
                raise ValueError("batch action decomposition returned mismatched option ids")
            return [
                ActionPlan(
                    option_id=option["option_id"],
                    description=option["action_description"],
                    physical_action=by_id[str(option["option_id"])].physical_action,
                    mental_action=by_id[str(option["option_id"])].mental_action,
                )
                for option in options
            ]
        except Exception:
            return await gather_limited(
                options,
                lambda option: self._decompose_one(option, stats),
                len(options),
            )

    async def _decompose_one(self, option: dict[str, Any], stats: RunStats) -> ActionPlan:
        reply = await self.llm.json_call(
            prompt=prompts.single_action_prompt(option["action_description"]),
            schema=ActionPlan,
        )
        stats.add(reply)
        data = reply.data
        return ActionPlan(
            option_id=option["option_id"],
            description=option["action_description"],
            physical_action=data.get("physical_action", ""),
            mental_action=data.get("mental_action", ""),
        )

    async def _simulate_branch(self, state: WorldState, plan: ActionPlan, stats: RunStats) -> Branch:
        branch = Branch(plan)
        try:
            physical_reply = await self.llm.json_call(
                prompt=prompts.physical_transition_prompt(
                    state.physical_state.as_dict(),
                    state.mental_state.as_dict(),
                    {"option_id": plan.option_id, "physical_action": plan.physical_action},
                ),
                schema=PhysicalState,
            )
            stats.add(physical_reply)
            branch.physical = PhysicalState.model_validate(physical_reply.data)
            mental_reply = await self.llm.json_call(
                prompt=prompts.mental_transition_prompt(
                    state.physical_state.as_dict(),
                    state.mental_state.as_dict(),
                    plan.as_dict(),
                    branch.physical.as_dict(),
                ),
                schema=MentalState,
            )
            stats.add(mental_reply)
            branch.mental = MentalState.model_validate(mental_reply.data)
            branch.successor = WorldState(physical_state=branch.physical, mental_state=branch.mental)
        except Exception as exc:
            branch.error = f"{type(exc).__name__}: {exc}"
        return branch

    async def _score_branches(
        self,
        sample: Sample,
        state: WorldState,
        observation: Observation,
        branches: list[Branch],
        stats: RunStats,
    ) -> None:
        alive = [b for b in branches if not b.failed and b.successor is not None]
        if not alive:
            return
        payloads = [
            {
                "option_id": b.plan.option_id,
                "action": b.plan.as_dict(),
                "next_state": b.successor.as_dict(),
            }
            for b in alive
        ]
        try:
            reply = await self.llm.json_call(
                prompt=prompts.scores_prompt(
                    sample.target_agent,
                    state.as_dict(),
                    normalize_observation(observation.as_dict()),
                    sample.question,
                    payloads,
                ),
                schema=BranchScoreBatch,
            )
            stats.add(reply)
            batch = BranchScoreBatch.model_validate(reply.data)
        except Exception as exc:
            for branch in alive:
                branch.error = f"scoring failed: {type(exc).__name__}: {exc}"
            return
        by_id = {str(score.option_id): score for score in batch.scores}
        for branch in alive:
            score = by_id.get(str(branch.plan.option_id))
            if score is None:
                branch.error = "scoring returned no entry for this option"
                continue
            score.option_id = branch.plan.option_id
            score.weighted_score = weighted_score(score, self.settings.score_weights)
            branch.score = score


def weighted_score(score: BranchScore, weights: dict[str, float]) -> float:
    if score.safety_veto:
        return 0.0
    return round(
        float(weights.get("mentally_consistent", 0.0)) * score.mentally_consistent
        + float(weights.get("physically_plausible", 0.0)) * score.physically_plausible
        + float(weights.get("socially_appropriate", 0.0)) * score.socially_appropriate,
        4,
    )


def select_action(
    scores: list[BranchScore], weights: dict[str, float], sample_id: str
) -> tuple[str, dict[str, Any]]:
    def sort_key(score: BranchScore) -> tuple[float, float, float]:
        rank = -float(score.relative_rank) if score.relative_rank else -99.0
        return (score.weighted_score, rank, seeded_value(sample_id, score.option_id))

    ordered = sorted(scores, key=sort_key, reverse=True)
    winner = ordered[0]
    tied = [s.option_id for s in ordered if s.weighted_score == winner.weighted_score]
    return winner.option_id, {
        "weights": dict(weights),
        "tie_between": tied if len(tied) > 1 else [],
        "ranking": [
            {
                "option_id": s.option_id,
                "weighted_score": s.weighted_score,
                "relative_rank": s.relative_rank,
                "safety_veto": s.safety_veto,
            }
            for s in ordered
        ],
    }


def seeded_value(sample_id: str, option_id: str) -> float:
    digest = hashlib.sha256(f"{sample_id}|{option_id}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def public_record(record: dict[str, Any]) -> dict[str, Any]:
    hidden = {"answer", "golden_answer"}
    return {
        key: value
        for key, value in record.items()
        if key not in hidden and not str(key).startswith("_")
    }
