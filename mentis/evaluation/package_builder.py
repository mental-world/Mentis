"""Shared prediction/gold packaging helpers for evaluation."""

from __future__ import annotations

from typing import Any

from mentis.schema import SampleInput
from mentis.utils.runtime_input import normalize_record, runtime_sample_from_input


def records_by_sample_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(normalize_record(record).get("sample_id")): normalize_record(record) for record in records}


def generated_results(pred_record: dict[str, Any]) -> dict[str, Any]:
    return dict(pred_record.get("generated_results", {}) or {})


def golden_answer(gold_record: dict[str, Any]) -> dict[str, Any]:
    return dict(gold_record.get("golden_answer", {}) or {})


def sample_context(record: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_record(record)
    try:
        runtime = runtime_sample_from_input(SampleInput.model_validate(normalized))
        keys = ["sample_id", "task_type", "scene", "target_agent", "question", "options"]
        return {key: runtime.get(key) for key in keys if key in runtime}
    except Exception:
        keys = ["sample_id", "scene", "target_agent", "question", "options"]
        return {key: normalized.get(key) for key in keys if key in normalized}


def item_package(
    pred_record: dict[str, Any],
    gold_record: dict[str, Any],
    item: str,
    option_id: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    generated = generated_results(pred_record)
    golden = golden_answer(gold_record)
    context = sample_context(gold_record or pred_record)
    pred_payload = {"sample_context": context, "predicted": {item: generated.get(item)}}
    gold_payload = {"sample_context": context, "gold": {item: golden.get(item)}}
    if item == "next_state_s_{t+1}":
        pred_payload, gold_payload = _next_state_package(
            generated, golden, context, item, option_id
        )
    if item == "score":
        pred_payload["predicted"]["final_action"] = generated.get("final_action")
        gold_payload["gold"]["final_action"] = golden.get("final_action")
    return pred_payload, gold_payload


def next_state_option_ids(generated: dict[str, Any], golden: dict[str, Any]) -> list[str]:
    return sorted(
        set(dict_keys(generated.get("next_state_s_{t+1}", {})))
        | set(dict_keys(golden.get("next_state_s_{t+1}", {})))
    )


def dict_keys(value: Any) -> list[str]:
    return sorted(str(key) for key in value) if isinstance(value, dict) else []


def _next_state_package(
    generated: dict[str, Any],
    golden: dict[str, Any],
    context: dict[str, Any],
    item: str,
    option_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pred_next = generated.get(item, {})
    gold_next = golden.get(item, {})
    pred_branch = pred_next.get(option_id) if isinstance(pred_next, dict) else None
    gold_branch = gold_next.get(option_id) if isinstance(gold_next, dict) else None
    pred_payload = {
        "sample_context": context,
        "predicted": {
            "item": item,
            "option_id": option_id,
            "next_state_branch": pred_branch,
            "current_state_s_t": generated.get("current_state_s_t"),
            "candidate_action": _action_for_option(
                generated.get("candidate_actions", []), option_id
            ),
        },
    }
    gold_payload = {
        "sample_context": context,
        "gold": {
            "item": item,
            "option_id": option_id,
            "next_state_branch": gold_branch,
            "current_state_s_t": golden.get("current_state_s_t"),
        },
    }
    return pred_payload, gold_payload


def _action_for_option(actions: Any, option_id: str) -> dict[str, Any] | None:
    if not isinstance(actions, list):
        return None
    for action in actions:
        if isinstance(action, dict) and str(action.get("option_id")) == option_id:
            return action
    return None
