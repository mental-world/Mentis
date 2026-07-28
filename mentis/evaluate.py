from __future__ import annotations

from collections import defaultdict
from typing import Any


def evaluate(
    predictions: list[dict[str, Any]], gold: list[dict[str, Any]]
) -> dict[str, Any]:
    gold_by_id = {str(r.get("sample_id")): r for r in gold}
    pred_by_id = {str(r.get("sample_id")): r for r in predictions}
    rows = []
    for sample_id, gold_record in gold_by_id.items():
        answer = str(gold_record.get("answer") or "").strip()
        pred_record = pred_by_id.get(sample_id)
        result = (pred_record or {}).get("result") or {}
        prediction = str(result.get("prediction") or "").strip()
        rows.append(
            {
                "sample_id": sample_id,
                "answer": answer,
                "prediction": prediction,
                "correct": bool(answer) and prediction == answer,
                "failed": pred_record is None or result.get("status") != "success",
                "modality": str(gold_record.get("modality") or "unknown"),
                "scene_category": str(gold_record.get("scene_category") or "unknown"),
                "domain_category": str(gold_record.get("domain_category") or "unknown"),
            }
        )
    report = {
        "num_samples": len(rows),
        "num_predictions": len(pred_by_id),
        "missing_predictions": sorted(set(gold_by_id) - set(pred_by_id)),
        "accuracy": _accuracy(rows),
        "macro_f1": _macro_f1(rows),
        "failure_rate": round(sum(r["failed"] for r in rows) / len(rows), 4) if rows else 0.0,
        "per_modality": _grouped(rows, "modality"),
        "per_scene_category": _grouped(rows, "scene_category"),
        "per_domain_category": _grouped(rows, "domain_category"),
        "per_sample": rows,
    }
    return report


def _accuracy(rows: list[dict[str, Any]]) -> float:
    return round(sum(r["correct"] for r in rows) / len(rows), 4) if rows else 0.0


def _macro_f1(rows: list[dict[str, Any]]) -> float:
    labels = sorted({r["answer"] for r in rows if r["answer"]})
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        tp = sum(1 for r in rows if r["prediction"] == label and r["answer"] == label)
        fp = sum(1 for r in rows if r["prediction"] == label and r["answer"] != label)
        fn = sum(1 for r in rows if r["prediction"] != label and r["answer"] == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(f1)
    return round(sum(scores) / len(scores), 4)


def _grouped(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return {
        name: {
            "count": len(items),
            "accuracy": _accuracy(items),
            "macro_f1": _macro_f1(items),
        }
        for name, items in sorted(groups.items())
    }


def format_report(report: dict[str, Any]) -> str:
    lines = [
        f"samples={report['num_samples']} predictions={report['num_predictions']} "
        f"accuracy={report['accuracy']:.4f} macro_f1={report['macro_f1']:.4f} "
        f"failure_rate={report['failure_rate']:.4f}"
    ]
    for group_key in ("per_modality", "per_scene_category"):
        for name, stats in report[group_key].items():
            lines.append(
                f"  {group_key[4:]}={name}: n={stats['count']} "
                f"acc={stats['accuracy']:.4f} f1={stats['macro_f1']:.4f}"
            )
    if report["missing_predictions"]:
        lines.append(f"  missing predictions: {len(report['missing_predictions'])}")
    return "\n".join(lines)
