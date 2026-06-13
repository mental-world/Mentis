"""CLI wrapper for Mentis evaluation."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone

from mentis.config import load_config
from mentis.evaluation.judge import add_llm_judge_report
from mentis.evaluation.metrics import evaluate_predictions
from mentis.utils.json_utils import read_records, save_json
from mentis.utils.tracing import build_readable_run_id, sample_scope_label


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Mentis outputs")
    parser.add_argument("--pred", required=True, help="Predicted JSON/JSONL path")
    parser.add_argument("--gold", required=True, help="Gold JSON/JSONL path")
    parser.add_argument("--output", required=True, help="Report JSON path")
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
    run_manifest = _evaluation_manifest(args, config.models.judge_model, pred, gold)
    report = await evaluate_records(
        pred,
        gold,
        config_path=args.config,
        skip_llm_judge=args.skip_llm_judge,
        run_id=run_manifest["run_id"],
        run_manifest=run_manifest,
    )
    save_json(report, args.output)


def _evaluation_manifest(
    args: argparse.Namespace,
    judge_model: str,
    pred: list[dict],
    gold: list[dict],
) -> dict:
    created_at = datetime.now(timezone.utc).astimezone()
    sample_ids = [str(record.get("sample_id")) for record in pred if record.get("sample_id") is not None]
    gold_ids = {str(record.get("sample_id")) for record in gold if record.get("sample_id") is not None}
    is_all = bool(sample_ids) and set(sample_ids) == gold_ids
    scope = "all" if is_all else sample_scope_label(sample_ids)
    run_id = build_readable_run_id(
        run_type="eval",
        mode="judge",
        model=judge_model,
        sample_scope=scope,
        created_at=created_at.replace(tzinfo=None),
    )
    return {
        "run_id": run_id,
        "run_type": "evaluation",
        "mode": "judge",
        "model": judge_model,
        "sample_scope": "all" if is_all else "selected",
        "sample_count": len(pred),
        "sample_ids": None if is_all else sorted(set(sample_ids), key=_sample_sort_key),
        "pred": str(args.pred),
        "gold": str(args.gold),
        "output": str(args.output),
        "config": str(args.config),
        "skip_llm_judge": bool(args.skip_llm_judge),
        "created_at": created_at.isoformat(timespec="seconds"),
    }


def _sample_sort_key(sample_id: str) -> tuple[int, object]:
    return (0, int(sample_id)) if sample_id.isdigit() else (1, sample_id)


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(run_async(args))


if __name__ == "__main__":
    main()
