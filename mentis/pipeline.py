"""End-to-end Mentis pipeline."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from mentis.clients import create_llm_client
from mentis.config import MentisConfig
from mentis.modules.action_parser import ActionParser
from mentis.modules.decision import DecisionModule
from mentis.modules.direct_answer_baseline import DirectAnswerBaseline
from mentis.modules.mental_transition import MentalTransitionModel
from mentis.modules.next_state_merger import NextStateMerger
from mentis.modules.observation_generator import ObservationGenerator
from mentis.modules.physical_transition import PhysicalTransitionModel
from mentis.modules.scoring import ScoringModule
from mentis.modules.state_parser import StateParser
from mentis.modules.target_pseudo_agent import TargetPseudoAgent
from mentis.policies.ablation import AblationPolicy, get_ablation_policy
from mentis.schema import (
    CandidateAction,
    GeneratedResults,
    OptionInput,
    MentalState,
    PhysicalState,
    SampleInput,
    ScoreResult,
    TargetObservation,
    WorldState,
    normalize_target_observation,
)
from mentis.utils.concurrency_utils import gather_bounded
from mentis.utils.tracing import RunLogger, sum_token_usage
from mentis.utils.runtime_input import normalize_record, runtime_sample_from_input


@dataclass(frozen=True)
class RunContext:
    sample_id: str
    task_type: str
    question: str
    target_agent_description: str
    media_scene: dict[str, Any]
    input_base_dir: str = ""

    @classmethod
    def from_sample(cls, sample: dict[str, Any], input_base_dir: str = "") -> "RunContext":
        media_scene = sample.get("_media_scene") if isinstance(sample.get("_media_scene"), dict) else {}
        return cls(
            sample_id=str(sample.get("sample_id", "")),
            task_type=str(sample.get("task_type") or sample.get("scene") or sample.get("domain") or ""),
            question=str(sample.get("question") or ""),
            target_agent_description=str(
                sample.get("target_agent_description") or sample.get("target_agent") or ""
            ),
            media_scene=dict(media_scene),
            input_base_dir=input_base_dir,
        )


@dataclass
class BranchResult:
    action: CandidateAction
    physical: PhysicalState | None = None
    mental: MentalState | None = None
    next_state: WorldState | None = None
    score: ScoreResult | None = None
    status: str = "success"
    error_type: str | None = None
    error_message: str | None = None
    metadata: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.status == "failed"

    def mark_failed(self, exc: BaseException) -> None:
        self.status = "failed"
        self.error_type = type(exc).__name__
        self.error_message = str(exc)
        self.metadata["error"] = {
            "option_id": self.action.option_id,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "call_count": 0,
            "warnings": [str(exc)],
        }


class MentisPipeline:
    def __init__(
        self,
        config: MentisConfig,
        *,
        run_id: str = "",
        run_manifest: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.client = create_llm_client(config)
        logger = RunLogger(
            config.logging.output_dir,
            config.logging.save_raw_llm_outputs or config.logging.save_prompts,
            run_id=run_id,
            manifest=run_manifest,
        )
        self.run_id = logger.run_id
        self.state_parser = StateParser(self.client, config, logger)
        self.observation_generator = ObservationGenerator(self.client, config, logger)
        self.target_pseudo_agent = TargetPseudoAgent(config.transition.expected_action_count)
        self.action_parser = ActionParser(self.client, config, logger)
        self.physical_transition = PhysicalTransitionModel(self.client, config, logger)
        self.mental_transition = MentalTransitionModel(self.client, config, logger)
        self.merger = NextStateMerger()
        self.scoring = ScoringModule(self.client, config, logger)
        self.decision = DecisionModule(config)
        self.direct_answer = DirectAnswerBaseline(self.client, config, logger)

    async def run_record(
        self,
        record: dict[str, Any],
        *,
        ablation: str = "full_mwm",
        input_base_dir: str = "",
    ) -> dict[str, Any]:
        sample_record = normalize_record(record)
        output_record = _public_output_record(record)
        try:
            policy = get_ablation_policy(ablation)
            sample = SampleInput.model_validate(sample_record)
            media_base_dir = input_base_dir or str(record.get("_input_base_dir", ""))
            runtime_sample = runtime_sample_from_input(sample, media_base_dir)
            generated = await self._run_sample(
                runtime_sample,
                context=RunContext.from_sample(runtime_sample, media_base_dir),
                policy=policy,
            )
        except Exception as exc:
            generated = GeneratedResults.failed(
                error_type=type(exc).__name__,
                error_message=str(exc),
                metadata={"ablation": ablation, "run_id": self.run_id},
            )
        output_record["generated_results"] = generated.as_dict()
        return output_record

    async def _run_sample(
        self, sample: dict[str, Any], *, context: RunContext, policy: AblationPolicy
    ) -> GeneratedResults:
        if policy.direct_answer:
            return await self._run_direct_baseline(sample, policy)
        result = GeneratedResults(status="failed")
        module_meta: dict[str, dict[str, Any]] = {}
        branch_outputs: list[BranchResult] = []

        try:
            world_state, state_meta = await self._get_world_state(sample, context, policy)
            module_meta["state_parser"] = state_meta
            result.current_state_s_t = world_state.as_state_dict()

            observation, observation_meta = await self._get_observation(
                sample,
                context,
                policy,
                world_state,
            )
            module_meta["observation"] = observation_meta
            result.target_agent_observation_o_t = normalize_target_observation(
                observation.as_observation_dict()
            )

            options = [OptionInput.model_validate(opt) for opt in sample.get("options", [])]
            sampled_actions, pseudo_agent_meta = self.target_pseudo_agent.sample_actions(
                options=options,
            )
            module_meta["target_pseudo_agent"] = pseudo_agent_meta

            actions, action_meta = await self.action_parser.parse(
                sample_id=context.sample_id,
                sampled_actions=sampled_actions,
            )
            module_meta["action_parser"] = action_meta
            result.candidate_actions = [action.as_dict() for action in actions]

            branch_outputs = await self._run_branches(
                context=context,
                world_state=world_state,
                observation=observation,
                actions=actions,
                policy=policy,
            )
            result.next_state_s_t1 = _next_states_from_branches(branch_outputs)
            result.score = _score_table_from_branches(branch_outputs)

            failed_branches = _failed_branches(branch_outputs)
            if failed_branches:
                result.error_type = "BranchExecutionError"
                result.error_message = _branch_error_message(failed_branches)
                result.metadata = self._metadata(policy, module_meta, branch_outputs)
                return result

            scores = _scores_from_branches(branch_outputs)
            final_action, decision_trace = self.decision.decide(scores)
            result.status = "success"
            result.final_action = final_action
            result.decision_trace = decision_trace
            result.metadata = self._metadata(policy, module_meta, branch_outputs)
            return result
        except Exception as exc:
            result.error_type = type(exc).__name__
            result.error_message = str(exc)
            result.metadata = self._metadata(policy, module_meta, branch_outputs)
            return result

    async def _run_direct_baseline(
        self, sample: dict[str, Any], policy: AblationPolicy
    ) -> GeneratedResults:
        final, meta = await self.direct_answer.answer(sample)
        return GeneratedResults(
            status="success",
            final_action=final,
            decision_trace={"mode": "direct_answer_baseline", "metadata": meta},
            metadata=self._metadata(policy, {"direct_answer": meta}, []),
        )

    async def _get_world_state(
        self, sample: dict[str, Any], context: RunContext, policy: AblationPolicy
    ) -> tuple[WorldState, dict]:
        golden = sample.get("golden_answer") or {}
        if policy.use_oracle_state:
            oracle = golden.get("current_state_s_t")
            if not oracle:
                raise ValueError("oracle_state requires golden_answer.current_state_s_t")
            return WorldState.model_validate(oracle), {"source": "oracle", "call_count": 0}
        state_input, media_scene = build_state_parser_input(sample)
        return await self.state_parser.parse(
            sample_id=context.sample_id,
            state_input=state_input,
            media_scene=media_scene,
        )

    async def _get_observation(
        self,
        sample: dict[str, Any],
        context: RunContext,
        policy: AblationPolicy,
        world_state: WorldState,
    ) -> tuple[TargetObservation, dict]:
        golden = sample.get("golden_answer") or {}
        if policy.use_oracle_observation:
            oracle = golden.get("target_agent_observation_o_t")
            if not oracle:
                raise ValueError(
                    "oracle_observation requires golden_answer.target_agent_observation_o_t"
                )
            return TargetObservation.model_validate(oracle), {"source": "oracle", "call_count": 0}
        return await self.observation_generator.generate(
            context.sample_id,
            world_state,
            context.target_agent_description,
        )

    async def _run_branches(
        self,
        *,
        context: RunContext,
        world_state: WorldState,
        observation: TargetObservation,
        actions: list[CandidateAction],
        policy: AblationPolicy,
    ) -> list[BranchResult]:
        async def worker(action: CandidateAction) -> BranchResult:
            return await self._run_one_branch(
                context,
                world_state,
                observation,
                action,
                policy,
            )

        return await gather_bounded(
            actions,
            worker,
            max(1, min(len(actions), self.config.transition.max_concurrency)),
        )

    async def _run_one_branch(
        self,
        context: RunContext,
        world_state: WorldState,
        observation: TargetObservation,
        action: CandidateAction,
        policy: AblationPolicy,
    ) -> BranchResult:
        branch = BranchResult(action)
        physical_result, mental_result = await asyncio.gather(
            self.physical_transition.predict(context.sample_id, world_state, action),
            self.mental_transition.predict(context.sample_id, world_state, action),
            return_exceptions=True,
        )
        if isinstance(physical_result, BaseException):
            branch.mark_failed(physical_result)
        else:
            physical, physical_meta = physical_result
            branch.physical = physical
            branch.metadata["physical_transition"] = physical_meta
        if isinstance(mental_result, BaseException):
            branch.mark_failed(mental_result)
        else:
            mental, mental_meta = mental_result
            branch.mental = mental
            branch.metadata["mental_transition"] = mental_meta
        if branch.failed:
            return branch
        if branch.physical is None or branch.mental is None:
            branch.mark_failed(RuntimeError("Branch transition did not return both physical and mental states"))
            return branch

        next_state = self.merger.merge(branch.physical, branch.mental)
        branch.next_state = next_state
        try:
            score, score_meta = await self.scoring.score(
                sample_id=context.sample_id,
                world_state=world_state,
                observation=observation,
                question=context.question,
                target_agent=context.target_agent_description,
                action=action,
                next_state=next_state,
                ablation=policy,
            )
        except Exception as exc:
            branch.mark_failed(exc)
            return branch
        branch.score = score
        branch.metadata["scoring"] = score_meta
        return branch

    def _metadata(
        self,
        policy: AblationPolicy,
        module_meta: dict[str, dict[str, Any]],
        branch_outputs: list[BranchResult],
    ) -> dict[str, Any]:
        branch_meta = {
            branch.action.option_id: branch.metadata for branch in branch_outputs
        }
        all_meta = list(module_meta.values())
        for branch in branch_outputs:
            all_meta.extend(branch.metadata.values())
        token_usage = _metadata_field_summary(module_meta, branch_meta, "token_usage")
        total_token_usage = sum_token_usage(meta.get("token_usage", {}) for meta in all_meta)
        if total_token_usage:
            token_usage["total"] = total_token_usage
        return {
            "ablation": policy.name,
            "run_id": self.run_id,
            "module_metadata": {
                **module_meta,
                "branches": branch_meta,
            },
            "models": _metadata_field_summary(module_meta, branch_meta, "model"),
            "llm_call_count": _metadata_call_count(all_meta),
            "latency_ms": _metadata_field_summary(module_meta, branch_meta, "latency_ms"),
            "token_usage": token_usage,
            "requests": _metadata_field_summary(module_meta, branch_meta, "request"),
            "warnings": _metadata_warnings(all_meta),
        }


def _metadata_field_summary(
    module_meta: dict[str, dict[str, Any]],
    branch_meta: dict[str, dict[str, dict[str, Any]]],
    field: str,
) -> dict[str, Any]:
    summary = {
        name: meta[field]
        for name, meta in module_meta.items()
        if field in meta and meta[field]
    }
    branch_summary = {
        option_id: {
            name: meta[field]
            for name, meta in option_meta.items()
            if field in meta and meta[field]
        }
        for option_id, option_meta in branch_meta.items()
    }
    branch_summary = {
        option_id: fields for option_id, fields in branch_summary.items() if fields
    }
    if branch_summary:
        summary["branches"] = branch_summary
    return summary


def _metadata_call_count(items: list[dict[str, Any]]) -> int:
    return sum(
        int(meta.get("call_count", 1 if meta.get("model") else 0))
        for meta in items
    )


def _metadata_warnings(items: list[dict[str, Any]]) -> list[str]:
    warnings = []
    for meta in items:
        warnings.extend(meta.get("warnings", []))
    return sorted(set(warnings))


def _next_states_from_branches(branch_outputs: list[BranchResult]) -> dict[str, Any]:
    return {
        branch.action.option_id: branch.next_state.as_state_dict()
        for branch in branch_outputs
        if branch.next_state is not None
    }


def _score_table_from_branches(branch_outputs: list[BranchResult]) -> dict[str, Any]:
    return {
        branch.score.option_id: branch.score.as_score_dict()
        for branch in branch_outputs
        if branch.score is not None
    }


def _failed_branches(branch_outputs: list[BranchResult]) -> list[BranchResult]:
    return [branch for branch in branch_outputs if branch.failed]


def _scores_from_branches(branch_outputs: list[BranchResult]) -> list[ScoreResult]:
    return [branch.score for branch in branch_outputs if branch.score is not None]


def _branch_error_message(failed_branches: list[BranchResult]) -> str:
    parts = []
    for branch in failed_branches:
        option_id = branch.action.option_id
        error_type = branch.error_type or "Error"
        error_message = branch.error_message or ""
        parts.append(f"{option_id}: {error_type}: {error_message}")
    return "; ".join(parts)


def build_state_parser_input(sample: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    story = sample.get("story") if isinstance(sample.get("story"), dict) else {}
    key, value = _story_modality_item(story)
    state_input = {"story": {key: value}}
    media_scene = _state_media_scene(sample.get("_media_scene") or {}, key)
    return state_input, media_scene


def _story_modality_item(story: dict[str, Any]) -> tuple[str, Any]:
    for key in ["images", "video", "text"]:
        value = story.get(key)
        if _has_story_modality_value(value):
            return key, value
    raise ValueError(
        "State parser input must contain exactly one story modality: "
        "story.text, story.images, or story.video"
    )


def _state_media_scene(media_scene: dict[str, Any], story_key: str) -> dict[str, Any]:
    if story_key == "images":
        return {
            "image_paths": list(media_scene.get("image_paths") or []),
            "modality": "image",
        }
    if story_key == "video":
        return {
            "video_path": media_scene.get("video_path") or "",
            "modality": "video",
        }
    return {"modality": "text"}


def _has_story_modality_value(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return value is not None


def _public_output_record(record: dict[str, Any]) -> dict[str, Any]:
    hidden_keys = {"golden_answer"}
    return {
        key: value
        for key, value in record.items()
        if not key.startswith("_") and key not in hidden_keys
    }
