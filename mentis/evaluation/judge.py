"""LLM-based semantic judges for Mentis evaluation."""

from __future__ import annotations

from statistics import mean
from typing import Any, Callable

from mentis.clients import create_llm_client
from mentis.config import MentisConfig
from mentis.modules.base import ModuleBase
from mentis.prompts import (
    build_next_state_judge_prompt,
    build_observation_judge_prompt,
    build_score_judge_prompt,
    build_state_judge_prompt,
)
from mentis.schema import JudgeResult
from mentis.evaluation.package_builder import (
    generated_results,
    golden_answer,
    item_package,
    next_state_option_ids,
    records_by_sample_id,
)
from mentis.utils.concurrency_utils import gather_bounded
from mentis.utils.tracing import RunLogger, response_metadata


PromptBuilder = Callable[[dict[str, Any], dict[str, Any]], str]

JUDGE_ITEMS: dict[str, tuple[str, PromptBuilder]] = {
    "current_state_s_t": ("judge_current_state", build_state_judge_prompt),
    "target_agent_observation_o_t": ("judge_observation", build_observation_judge_prompt),
    "next_state_s_{t+1}": ("judge_next_state", build_next_state_judge_prompt),
    "score": ("judge_score", build_score_judge_prompt),
}


async def add_llm_judge_report(
    report: dict[str, Any],
    pred_records: list[dict[str, Any]],
    gold_records: list[dict[str, Any]],
    config: MentisConfig,
    run_id: str = "",
    run_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    judge_report = await judge_predictions(
        pred_records,
        gold_records,
        config,
        run_id=run_id,
        run_manifest=run_manifest,
    )
    merged = dict(report)
    merged["llm_judge"] = judge_report
    _attach_per_sample_judges(merged, judge_report)
    _replace_top_level_judge_scores(merged, judge_report)
    return merged


async def judge_predictions(
    pred_records: list[dict[str, Any]],
    gold_records: list[dict[str, Any]],
    config: MentisConfig,
    *,
    run_id: str = "",
    run_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    client = create_llm_client(config)
    logger = RunLogger(
        config.logging.output_dir,
        config.logging.save_raw_llm_outputs or config.logging.save_prompts,
        run_id=run_id,
        manifest=run_manifest,
    )
    judge = LLMJudge(client, config, logger)
    jobs = _build_jobs(pred_records, gold_records)
    results = await gather_bounded(jobs, judge.judge_item, config.evaluation.judge_max_concurrency)
    return _build_report(results, config.models.judge_model)


class LLMJudge(ModuleBase):
    async def judge_item(self, job: dict[str, Any]) -> dict[str, Any]:
        item = job["item"]
        task, prompt_builder = JUDGE_ITEMS[item]
        pred_package, gold_package = item_package(
            job["pred"], job["gold"], item, job.get("option_id", "")
        )
        prompt = prompt_builder(pred_package, gold_package)
        response = await self.call_llm(
            sample_id=job["sample_id"],
            task=task,
            prompt=prompt,
            model=self.config.models.judge_model,
            context={
                "sample_id": job["sample_id"],
                "item": item,
                "pred": pred_package,
                "gold": gold_package,
            },
            schema=JudgeResult,
        )
        parsed = _normalize_judge_result(response.parsed, item)
        metadata = response_metadata(response)
        return {
            "sample_id": job["sample_id"],
            "task_type": job.get("task_type", ""),
            "item": item,
            "option_id": job.get("option_id", ""),
            "status": "success",
            "judge": parsed,
            "metadata": metadata,
        }


def _build_jobs(
    pred_records: list[dict[str, Any]], gold_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    gold_by_id = records_by_sample_id(gold_records)
    jobs = []
    for pred in pred_records:
        sample_id = str(pred.get("sample_id"))
        gold = gold_by_id.get(sample_id, {})
        for item in JUDGE_ITEMS:
            if item == "next_state_s_{t+1}":
                jobs.extend(_next_state_jobs(sample_id, pred, gold))
            elif _has_judge_material(pred, gold, item):
                jobs.append(
                    {
                        "sample_id": sample_id,
                        "task_type": pred.get("task_type") or gold.get("task_type", ""),
                        "pred": pred,
                        "gold": gold,
                        "item": item,
                    }
                )
    return jobs


def _next_state_jobs(sample_id: str, pred: dict[str, Any], gold: dict[str, Any]) -> list[dict[str, Any]]:
    option_ids = next_state_option_ids(generated_results(pred), golden_answer(gold))
    return [
        {
            "sample_id": sample_id,
            "task_type": pred.get("task_type") or gold.get("task_type", ""),
            "pred": pred,
            "gold": gold,
            "item": "next_state_s_{t+1}",
            "option_id": option_id,
        }
        for option_id in option_ids
    ]


def _has_judge_material(pred: dict[str, Any], gold_record: dict[str, Any], item: str) -> bool:
    return bool(generated_results(pred).get(item) or golden_answer(gold_record).get(item))


def _build_report(results: list[dict[str, Any]], judge_model: str) -> dict[str, Any]:
    return {
        "enabled": True,
        "judge_model": judge_model,
        "num_judged_items": len(results),
        "summary": _summary(results),
        "per_sample": _per_sample(results),
        "failed_items": [r for r in results if r["status"] != "success"],
    }


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for item in JUDGE_ITEMS:
        item_results = [r["judge"] for r in results if r["item"] == item and r["status"] == "success"]
        summary[item] = {
            "physical_score": _mean_score(item_results, "physical_score"),
            "mental_score": _mean_score(item_results, "mental_score"),
            "semantic_consistency_score": _mean_score(item_results, "semantic_consistency_score"),
            "coupling_score": _mean_score(item_results, "coupling_score"),
            "transition_score": _mean_score(item_results, "transition_score"),
        }
    return summary


def _per_sample(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for result in results:
        sample_id = str(result["sample_id"])
        row = grouped.setdefault(
            sample_id,
            {"sample_id": sample_id, "task_type": result.get("task_type", ""), "items": {}},
        )
        entry = {
            "status": result["status"],
            "judge": result["judge"],
            "metadata": result["metadata"],
        }
        if result["item"] == "next_state_s_{t+1}" and result.get("option_id"):
            row["items"].setdefault(result["item"], {})[result["option_id"]] = entry
        else:
            row["items"][result["item"]] = entry
    return list(grouped.values())


def _attach_per_sample_judges(report: dict[str, Any], judge_report: dict[str, Any]) -> None:
    judge_by_id = {str(r["sample_id"]): r["items"] for r in judge_report.get("per_sample", [])}
    for row in report.get("per_sample", []):
        row["llm_judge"] = judge_by_id.get(str(row.get("sample_id")), {})


def _replace_top_level_judge_scores(report: dict[str, Any], judge_report: dict[str, Any]) -> None:
    summary = judge_report.get("summary", {})
    state = summary.get("current_state_s_t", {})
    observation = summary.get("target_agent_observation_o_t", {})
    next_state = summary.get("next_state_s_{t+1}", {})
    report["state_physical_judge_score"] = state.get("physical_score")
    report["state_mental_judge_score"] = state.get("mental_score")
    report["observation_physical_judge_score"] = observation.get("physical_score")
    report["observation_mental_judge_score"] = observation.get("mental_score")
    report["next_state_physical_judge_score"] = next_state.get("physical_score")
    report["next_state_mental_judge_score"] = next_state.get("mental_score")
    report["transition_reasonableness_score"] = next_state.get("transition_score")
    report["coupling_validity_score"] = _mean_values(
        [
            state.get("coupling_score"),
            observation.get("coupling_score"),
            next_state.get("coupling_score"),
        ]
    )


def _normalize_judge_result(parsed: dict[str, Any], item: str) -> dict[str, Any]:
    result = JudgeResult.model_validate(parsed).as_dict()
    result["item"] = result.get("item") or item
    for key in _score_keys():
        result[key] = _clamp_optional(result.get(key))
    result["option_scores"] = _normalize_option_scores(result.get("option_scores", {}))
    return result


def _normalize_option_scores(option_scores: dict[str, Any]) -> dict[str, Any]:
    normalized = {}
    for option_id, value in option_scores.items():
        if not isinstance(value, dict):
            continue
        normalized[option_id] = dict(value)
        for key in _score_keys():
            if key in normalized[option_id]:
                normalized[option_id][key] = _clamp_optional(normalized[option_id].get(key))
    return normalized


def _mean_score(results: list[dict[str, Any]], key: str) -> float | None:
    return _mean_values([r.get(key) for r in results])


def _mean_values(values: list[Any]) -> float | None:
    nums = [_clamp_optional(value) for value in values]
    valid = [value for value in nums if value is not None]
    return round(mean(valid), 4) if valid else None


def _clamp_optional(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except Exception:
        return None
    return max(0.0, min(1.0, number))


def _score_keys() -> list[str]:
    return [
        "physical_score",
        "mental_score",
        "semantic_consistency_score",
        "coupling_score",
        "transition_score",
        "confidence",
    ]
