"""CLI wrapper for Mentis evaluation."""

from __future__ import annotations

import argparse
import asyncio

from mentis.config import load_config
from mentis.evaluation.judge import add_llm_judge_report
from mentis.evaluation.metrics import evaluate_predictions
from mentis.utils.json_utils import read_records, save_json


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
) -> dict:
    config = load_config(config_path)
    report = evaluate_predictions(pred, gold)
    if config.evaluation.llm_judge_enabled and not skip_llm_judge:
        report = await add_llm_judge_report(report, pred, gold, config)
    return report


async def run_async(args: argparse.Namespace) -> None:
    pred, _ = read_records(args.pred)
    gold, _ = read_records(args.gold)
    report = await evaluate_records(
        pred,
        gold,
        config_path=args.config,
        skip_llm_judge=args.skip_llm_judge,
    )
    save_json(report, args.output)


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(run_async(args))


if __name__ == "__main__":
    main()
