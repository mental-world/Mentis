from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from mentis.baseline import run_direct_sample
from mentis.config import load_env_file, load_settings, require_api_key
from mentis.engine import MentisEngine
from mentis.evaluate import evaluate, format_report
from mentis.llm import LLMClient
from mentis.utils import append_jsonl, read_jsonl, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mentis: an inspectable mental world model baseline")
    sub = parser.add_subparsers(dest="command", required=True)

    predict = sub.add_parser("predict", help="Run predictions over a JSONL benchmark file")
    predict.add_argument("--input", required=True, help="Input JSONL file")
    predict.add_argument("--output-dir", default="outputs", help="Directory for run artifacts")
    predict.add_argument("--config", default=None, help="Optional YAML config path")
    predict.add_argument("--system", choices=["mentis", "direct"], default="mentis")
    predict.add_argument("--sample-id", action="append", default=[], help="Only run these sample ids")
    predict.add_argument("--limit", type=int, default=None, help="Run at most N samples")

    score = sub.add_parser("evaluate", help="Score a predictions file against gold answers")
    score.add_argument("--predictions", required=True, help="predictions.jsonl from predict")
    score.add_argument("--gold", required=True, help="Benchmark JSONL file with answer fields")
    score.add_argument("--output", default=None, help="Optional report JSON path")
    return parser


async def run_predict(args: argparse.Namespace) -> None:
    load_env_file()
    require_api_key()
    settings = load_settings(args.config)
    records = read_jsonl(args.input)
    wanted = {s for group in args.sample_id for s in str(group).split(",") if s}
    if wanted:
        records = [r for r in records if str(r.get("sample_id")) in wanted]
        missing = wanted - {str(r.get("sample_id")) for r in records}
        if missing:
            raise SystemExit(f"sample ids not found in input: {', '.join(sorted(missing))}")
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise SystemExit("No records to run.")

    base_dir = str(Path(args.input).resolve().parent)
    run_name = f"{args.system}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"

    engine = MentisEngine(settings)
    outputs = []
    print(f"Running {args.system} on {len(records)} sample(s) with model {settings.model}")
    for index, record in enumerate(records, start=1):
        sample_id = str(record.get("sample_id", ""))
        started = time.perf_counter()
        if args.system == "direct":
            output = await run_direct_sample(engine.llm, record, base_dir)
        else:
            output = await engine.run_sample(record, base_dir)
        outputs.append(output)
        append_jsonl(predictions_path, output)
        result = output.get("result", {})
        print(
            f"[{index}/{len(records)}] sample={sample_id} "
            f"status={result.get('status')} prediction={result.get('prediction') or '-'} "
            f"calls={result.get('stats', {}).get('llm_calls', 0)} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    print(f"Wrote {predictions_path}")
    if any(str(r.get("answer") or "").strip() for r in records):
        report = evaluate(outputs, records)
        save_json(run_dir / "report.json", report)
        print(format_report(report))
        print(f"Wrote {run_dir / 'report.json'}")


def run_evaluate(args: argparse.Namespace) -> None:
    predictions = read_jsonl(args.predictions)
    gold = read_jsonl(args.gold)
    report = evaluate(predictions, gold)
    print(format_report(report))
    if args.output:
        save_json(args.output, report)
        print(f"Wrote {args.output}")


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "predict":
        asyncio.run(run_predict(args))
    else:
        run_evaluate(args)


if __name__ == "__main__":
    main()
