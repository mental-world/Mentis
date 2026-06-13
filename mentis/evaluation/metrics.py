"""Deterministic metrics for Mentis outputs."""

from __future__ import annotations

from statistics import mean
from typing import Any

from mentis.evaluation.package_builder import (
    generated_results,
    golden_answer,
    records_by_sample_id,
)


def evaluate_predictions(pred_records: list[dict[str, Any]], gold_records: list[dict[str, Any]]) -> dict[str, Any]:
    gold_by_id = records_by_sample_id(gold_records)
    rows = []
    for pred in pred_records:
        sample_id = str(pred.get("sample_id"))
        gold = gold_by_id.get(sample_id, {})
        rows.append(_evaluate_one(pred, gold))
    return {
        "num_samples": len(rows),
        "final_action_accuracy": _mean([r["final_action_correct"] for r in rows]),
        "failure_rate": _mean([r["failed"] for r in rows]),
        "score_alignment": {
            "mental_mae": _mean_non_null([r["mental_mae"] for r in rows]),
            "physical_mae": _mean_non_null([r["physical_mae"] for r in rows]),
            "social_mae": _mean_non_null([r["social_mae"] for r in rows]),
        },
        "per_sample": rows,
        "per_task_type": _group_by_task(rows),
    }


def _evaluate_one(pred: dict[str, Any], gold_record: dict[str, Any]) -> dict[str, Any]:
    generated = generated_results(pred)
    golden = golden_answer(gold_record)
    pred_final = generated.get("final_action", "")
    gold_final = golden.get("final_action", "")
    score_errors = _score_errors(generated.get("score", {}), golden.get("score", {}))
    next_state_stats = _next_state_stats(
        generated.get("next_state_s_{t+1}", {}),
        golden.get("next_state_s_{t+1}", {}),
    )
    return {
        "sample_id": pred.get("sample_id"),
        "task_type": _task_type(pred, gold_record),
        "failed": 1.0 if generated.get("status") == "failed" else 0.0,
        "pred_final_action": pred_final,
        "gold_final_action": gold_final,
        "final_action_correct": 1.0 if pred_final == gold_final and gold_final else 0.0,
        "has_current_state": bool(generated.get("current_state_s_t")),
        "has_observation": bool(generated.get("target_agent_observation_o_t")),
        "has_next_state": next_state_stats["schema_valid"],
        **next_state_stats,
        **score_errors,
    }


def _task_type(pred: dict[str, Any], gold_record: dict[str, Any]) -> str:
    return (
        str(pred.get("task_type") or "")
        or str(gold_record.get("task_type") or "")
        or str(pred.get("scene") or "")
        or str(gold_record.get("scene") or "")
        or str(pred.get("domain") or "")
        or str(gold_record.get("domain") or "")
        or "unknown"
    )


def _score_errors(pred_scores: dict[str, Any], gold_scores: dict[str, Any]) -> dict[str, Any]:
    mental = []
    physical = []
    social = []
    for option_id, gold in gold_scores.items():
        pred = pred_scores.get(option_id, {})
        if not pred:
            continue
        mental.append(abs(_num(pred.get("mentally_consistent")) - _num(gold.get("mentally_consistent"))))
        physical.append(abs(_num(pred.get("physically_plausible")) - _num(gold.get("physically_plausible"))))
        social.append(abs(_num(pred.get("socially_appropriate")) - _num(gold.get("socially_appropriate"))))
    return {
        "mental_mae": _mean_non_null(mental),
        "physical_mae": _mean_non_null(physical),
        "social_mae": _mean_non_null(social),
    }


def _next_state_stats(pred_next: dict[str, Any], gold_next: dict[str, Any]) -> dict[str, Any]:
    expected = set(gold_next) if isinstance(gold_next, dict) else set()
    predicted = set(pred_next) if isinstance(pred_next, dict) else set()
    matched = expected & predicted
    valid_options = [
        option_id
        for option_id in matched
        if _valid_next_state_branch(pred_next.get(option_id))
    ]
    has_all_expected = bool(expected) and matched == expected
    branches_schema_valid = bool(expected) and len(valid_options) == len(expected)
    return {
        "next_state_expected_options": sorted(expected),
        "next_state_predicted_options": sorted(predicted),
        "next_state_missing_options": sorted(expected - predicted),
        "next_state_extra_options": sorted(predicted - expected),
        "schema_valid": has_all_expected and branches_schema_valid,
    }


def _valid_next_state_branch(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("physical_state"), dict)
        and isinstance(value.get("mental_state"), dict)
    )


def _num(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _mean(values: list[float]) -> float:
    return round(mean(values), 4) if values else 0.0


def _mean_non_null(values: list[Any]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return round(mean(vals), 4) if vals else None


def _group_by_task(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row.get("task_type") or "unknown", []).append(row)
    return {
        task: {
            "count": len(items),
            "final_action_accuracy": _mean([x["final_action_correct"] for x in items]),
            "failure_rate": _mean([x["failed"] for x in items]),
        }
        for task, items in grouped.items()
    }
