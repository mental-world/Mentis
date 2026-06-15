"""CLI wrapper for Mentis evaluation."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path

from mentis.config import load_config
from mentis.evaluation.judge import add_llm_judge_report
from mentis.evaluation.metrics import evaluate_predictions
from mentis.utils.json_utils import read_records, save_json
from mentis.utils.tracing import build_readable_run_id, sample_scope_label


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Mentis outputs")
    parser.add_argument("--pred", required=True, help="Predicted JSON/JSONL path")
    parser.add_argument("--gold", required=True, help="Gold JSON/JSONL path")
    parser.add_argument(
        "--output",
        default="outputs",
        help="Report JSON path or directory. Default: outputs, which writes {run_id}_report.json",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Config YAML path")
    parser.add_argument(
        "--skip-llm-judge",
        action="store_true",
        help="Only compute deterministic metrics, without GPT judge calls",
    )
    return parser


async def evaluate_records(
    pred: list[dict],
    gold: list[dict],
    *,
    config_path: str = "configs/default.yaml",
    skip_llm_judge: bool = False,
    run_id: str = "",
    run_manifest: dict | None = None,
) -> dict:
    config = load_config(config_path)
    report = evaluate_predictions(pred, gold)
    if config.evaluation.llm_judge_enabled and not skip_llm_judge:
        report = await add_llm_judge_report(
            report,
            pred,
            gold,
            config,
            run_id=run_id,
            run_manifest=run_manifest,
        )
    return report


async def run_async(args: argparse.Namespace) -> None:
    pred, _ = read_records(args.pred)
    gold, _ = read_records(args.gold)
    config = load_config(args.config)
    created_at = datetime.now(timezone.utc).astimezone()
    run_id = build_evaluation_run_id(config.models.judge_model, pred, gold, created_at)
    output_path = resolve_report_output_path(args.output, run_id)
    run_manifest = _evaluation_manifest(
        args,
        config.models.judge_model,
        pred,
        gold,
        run_id=run_id,
        output_path=output_path,
        created_at=created_at,
    )
    report = await evaluate_records(
        pred,
        gold,
        config_path=args.config,
        skip_llm_judge=args.skip_llm_judge,
        run_id=run_manifest["run_id"],
        run_manifest=run_manifest,
    )
    save_json(report, output_path)
    print(f"Wrote evaluation report to {output_path}")


def resolve_report_output_path(output: str, run_id: str) -> Path:
    requested = Path(output)
    if requested.suffix.lower() == ".json":
        if requested.stem.lower() in {"eval_report", "evaluation_report", "report"}:
            return requested.with_name(f"{run_id}_report.json")
        return requested
    return requested / f"{run_id}_report.json"


def build_evaluation_run_id(
    judge_model: str,
    pred: list[dict],
    gold: list[dict],
    created_at: datetime,
) -> str:
    sample_ids = _prediction_sample_ids(pred)
    gold_ids = {str(record.get("sample_id")) for record in gold if record.get("sample_id") is not None}
    is_all = bool(sample_ids) and set(sample_ids) == gold_ids
    scope = "all" if is_all else sample_scope_label(sample_ids)
    ablation = _prediction_ablation_label(pred)
    return build_readable_run_id(
        run_type="eval",
        mode=f"{ablation}_judge",
        model=judge_model,
        sample_scope=scope,
        created_at=created_at.replace(tzinfo=None),
    )


def _evaluation_manifest(
    args: argparse.Namespace,
    judge_model: str,
    pred: list[dict],
    gold: list[dict],
    run_id: str,
    output_path: Path,
    created_at: datetime,
) -> dict:
    sample_ids = _prediction_sample_ids(pred)
    gold_ids = {str(record.get("sample_id")) for record in gold if record.get("sample_id") is not None}
    is_all = bool(sample_ids) and set(sample_ids) == gold_ids
    return {
        "run_id": run_id,
        "run_type": "evaluation",
        "mode": "judge",
        "prediction_ablation": _prediction_ablation_label(pred),
        "model": judge_model,
        "sample_scope": "all" if is_all else "selected",
        "sample_count": len(pred),
        "sample_ids": None if is_all else sorted(set(sample_ids), key=_sample_sort_key),
        "pred": str(args.pred),
        "gold": str(args.gold),
        "output": str(output_path),
        "config": str(args.config),
        "skip_llm_judge": bool(args.skip_llm_judge),
        "created_at": created_at.isoformat(timespec="seconds"),
    }


def _prediction_sample_ids(pred: list[dict]) -> list[str]:
    return [
        str(record.get("sample_id"))
        for record in pred
        if record.get("sample_id") is not None
    ]


def _prediction_ablation_label(pred: list[dict]) -> str:
    ablations = {
        str(metadata.get("ablation"))
        for metadata in (_generated_metadata(record) for record in pred)
        if metadata.get("ablation")
    }
    if len(ablations) == 1:
        return next(iter(ablations))
    if len(ablations) > 1:
        return "mixed_ablation"
    return "unknown_ablation"


def _generated_metadata(record: dict) -> dict:
    generated = record.get("generated_results")
    if not isinstance(generated, dict):
        return {}
    metadata = generated.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _sample_sort_key(sample_id: str) -> tuple[int, object]:
    return (0, int(sample_id)) if sample_id.isdigit() else (1, sample_id)


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(run_async(args))


if __name__ == "__main__":
    main()
