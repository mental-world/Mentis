"""CLI entry point for running Mentis."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import re

from mentis.config import load_config
from mentis.pipeline import MentisPipeline
from mentis.policies.ablation import ABLATION_CHOICES
from mentis.utils.json_utils import read_records


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
            "which writes {world_model}_predictions.jsonl. Generic "
            "predictions.* names are prefixed by world model."
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
    pipeline = MentisPipeline(config)
    records, _ = read_records(args.input)
    sample_ids = _sample_id_filter(args.sample_id)
    records = _filter_records_by_sample_ids(records, sample_ids)
    input_base_dir = str(Path(args.input).resolve().parent)
    output_path = resolve_output_path(args.output, config.models.world_model)
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


def resolve_output_path(output: str, world_model: str) -> Path:
    requested = Path(output)
    model_slug = model_filename_slug(world_model)
    if requested.suffix.lower() in {".json", ".jsonl"}:
        if requested.stem.lower() in {"prediction", "predictions"}:
            return requested.with_name(f"{model_slug}_{requested.name}")
        return requested
    return requested / f"{model_slug}_predictions.jsonl"


def model_filename_slug(model: str) -> str:
    name = model.strip()
    if not name:
        return "model"
    if name.lower().startswith("gpt-"):
        return "gpt" + "_".join(_slug_parts(name[4:]))
    return "_".join(_slug_parts(name))


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


def _slug_parts(text: str) -> list[str]:
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", text) if part]
    return [_format_slug_part(part) for part in parts] or ["MODEL"]


def _format_slug_part(part: str) -> str:
    return part.lower()


def main() -> None:
    args = build_arg_parser().parse_args()
    asyncio.run(run_async(args))


if __name__ == "__main__":
    main()
