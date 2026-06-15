"""CLI entry point for running Mentis."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from mentis.config import load_config
from mentis.pipeline import MentisPipeline
from mentis.policies.ablation import ABLATION_CHOICES
from mentis.utils.json_utils import read_records
from mentis.utils.tracing import build_readable_run_id, model_slug, sample_scope_label


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Mentis MWM pipeline")
    parser.add_argument(
        "--input",
        default="data/sample_input.jsonl",
        help="Input JSON or JSONL path. Default: data/sample_input.jsonl",
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help=(
            "Output JSON/JSONL path or directory. Default: outputs, "
            "which writes {run_id}_predictions.jsonl. Generic or legacy "
            "model_predictions.* names are replaced by the run-id filename."
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Config YAML path. Default: configs/default.yaml",
    )
    parser.add_argument(
        "--sample-id",
        action="append",
        nargs="+",
        default=[],
        help=(
            "Optional sample_id filter for JSONL inputs. "
            "Accepts one or more ids, and can be repeated."
        ),
    )
    parser.add_argument(
        "--ablation",
        default="full_mwm",
        choices=ABLATION_CHOICES,
    )
    return parser


async def run_async(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    records, _ = read_records(args.input)
    sample_ids = _sample_id_filter(args.sample_id)
    records = _filter_records_by_sample_ids(records, sample_ids)
    input_base_dir = str(Path(args.input).resolve().parent)
    created_at = datetime.now(timezone.utc).astimezone()
    run_id = build_prediction_run_id(
        args.ablation,
        config.models.world_model,
        sample_ids,
        created_at,
    )
    output_path = resolve_output_path(args.output, run_id, config.models.world_model)
    run_manifest = _prediction_manifest(
        args=args,
        config_model=config.models.world_model,
        output_path=output_path,
        sample_ids=sample_ids,
        sample_count=len(records),
        run_id=run_id,
        created_at=created_at,
    )
    pipeline = MentisPipeline(
        config,
        run_id=run_manifest["run_id"],
        run_manifest=run_manifest,
    )
    output_is_jsonl = output_path.suffix.lower() == ".jsonl"
    with StreamingRecordWriter(output_path, output_is_jsonl, len(records)) as writer:
        for record in records:
            output = await pipeline.run_record(
                record,
                ablation=args.ablation,
                input_base_dir=input_base_dir,
            )
            writer.write(output)
    print(f"Wrote predictions to {output_path}")


class StreamingRecordWriter:
    def __init__(self, path: Path, as_jsonl: bool, total_records: int) -> None:
        self.path = path
        self.as_jsonl = as_jsonl
        self.total_records = total_records
        self.file = None
        self.records_written = 0

    def __enter__(self) -> "StreamingRecordWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", encoding="utf-8")
        if not self.as_jsonl and self.total_records != 1:
            self.file.write("[\n")
        return self

    def write(self, record: dict) -> None:
        if self.file is None:
            raise RuntimeError("StreamingRecordWriter must be opened before writing")
        if self.as_jsonl:
            self.file.write(json.dumps(record, ensure_ascii=False) + "\n")
        elif self.total_records == 1:
            self.file.write(json.dumps(record, ensure_ascii=False, indent=2))
        else:
            prefix = "" if self.records_written == 0 else ",\n"
            self.file.write(prefix + json.dumps(record, ensure_ascii=False, indent=2))
        self.file.flush()
        self.records_written += 1

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.file is None:
            return
        if not self.as_jsonl and self.total_records != 1:
            self.file.write("\n]\n")
        elif not self.as_jsonl:
            self.file.write("\n")
        self.file.close()


def resolve_output_path(output: str, run_id: str, world_model: str) -> Path:
    requested = Path(output)
    if requested.suffix.lower() in {".json", ".jsonl"}:
        if _is_generic_prediction_name(requested.stem, world_model):
            return requested.with_name(f"{run_id}_predictions{requested.suffix}")
        return requested
    return requested / f"{run_id}_predictions.jsonl"


def _is_generic_prediction_name(stem: str, world_model: str) -> bool:
    normalized = stem.lower()
    return normalized in {
        "prediction",
        "predictions",
        f"{model_slug(world_model)}_predictions",
    }


def build_prediction_run_id(
    ablation: str,
    world_model: str,
    sample_ids: set[str],
    created_at: datetime,
) -> str:
    return build_readable_run_id(
        run_type="predict",
        mode=str(ablation),
        model=world_model,
        sample_scope=sample_scope_label(sample_ids),
        created_at=created_at.replace(tzinfo=None),
    )


def _sample_id_filter(values: list[list[str]]) -> set[str]:
    return {
        str(item)
        for group in values
        for item in group
        if str(item).strip()
    }


def _filter_records_by_sample_ids(
    records: list[dict], sample_ids: set[str]
) -> list[dict]:
    if not sample_ids:
        return records
    filtered = [
        record for record in records if str(record.get("sample_id")) in sample_ids
    ]
    found = {str(record.get("sample_id")) for record in filtered}
    missing = sorted(sample_ids - found)
    if missing:
        raise ValueError(f"No record found for sample_id(s): {', '.join(missing)}")
    return filtered


def _prediction_manifest(
    *,
    args: argparse.Namespace,
    config_model: str,
    output_path: Path,
    sample_ids: set[str],
    sample_count: int,
    run_id: str,
    created_at: datetime,
) -> dict:
    return {
        "run_id": run_id,
        "run_type": "prediction",
        "ablation": str(args.ablation),
        "model": config_model,
        "sample_scope": "all" if not sample_ids else "selected",
        "sample_count": sample_count,
        "sample_ids": None if not sample_ids else sorted(sample_ids, key=_sample_sort_key),
        "input": str(args.input),
        "output": str(output_path),
        "config": str(args.config),
        "created_at": created_at.isoformat(timespec="seconds"),
    }


def _sample_sort_key(sample_id: str) -> tuple[int, object]:
    return (0, int(sample_id)) if sample_id.isdigit() else (1, sample_id)


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(run_async(args))


if __name__ == "__main__":
    main()
