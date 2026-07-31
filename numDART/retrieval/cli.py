"""Command-line entry points for native DART CTA retrieval artifacts."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

from numDART.baselines.io import read_candidates_jsonl

from .dart_output import export_dart_candidates
from .ontology_input import build_dart_ontology
from .table_input import build_dart_column_inputs


@dataclass(frozen=True)
class CtaRetrievalConfig:
    """Configuration shared by preparation and result export."""

    run_id: str
    table_archive: Path
    ontology_source: Path
    dart_repo: Path
    model_path: Path
    output_dir: Path
    top_k: int = 20
    max_cells: int = 10
    max_parents: int = 3
    batch_size: int = 128
    device: str = "cuda"

    @classmethod
    def load(cls, path: str | Path) -> "CtaRetrievalConfig":
        """Loads a strict JSON configuration.

        Relative paths are resolved against the configuration directory.
        HPC-only paths need not exist during local preparation.
        """
        config_path = Path(path).resolve()
        value = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Configuration must be a JSON object")
        expected = set(cls.__dataclass_fields__)
        unknown = set(value) - expected
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        required = {"run_id", "table_archive", "ontology_source", "dart_repo",
                    "model_path", "output_dir"}
        missing = required - set(value)
        if missing:
            raise ValueError(f"Missing configuration fields: {sorted(missing)}")

        base = config_path.parent
        for key in ("table_archive", "ontology_source", "dart_repo", "model_path",
                    "output_dir"):
            value[key] = _resolve_path(base, str(value[key]))
        config = cls(**value)
        config._validate()
        return config

    def _validate(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        for name in ("top_k", "max_cells", "max_parents", "batch_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if not self.device.strip():
            raise ValueError("device must not be empty")


def main(argv: Sequence[str] | None = None) -> int:
    """Runs one artifact preparation, export, or validation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "export", "validate"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", required=True)
        if command == "validate":
            subparser.add_argument(
                "--stage", choices=("prepared", "complete"), default="complete"
            )
    arguments = parser.parse_args(argv)
    config = CtaRetrievalConfig.load(arguments.config)

    if arguments.command == "prepare":
        _prepare(config)
    elif arguments.command == "export":
        _export(config)
    else:
        _validate_outputs(config, arguments.stage)
    return 0


def _prepare(config: CtaRetrievalConfig) -> None:
    if not config.table_archive.is_file():
        raise FileNotFoundError(config.table_archive)
    if not config.ontology_source.is_file():
        raise FileNotFoundError(config.ontology_source)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_dart_column_inputs(
        config.table_archive,
        config.output_dir / "columns",
        config.output_dir / "manifest.jsonl",
        config.max_cells,
    )
    class_count = build_dart_ontology(
        config.ontology_source, config.output_dir / "ontology.json"
    )
    _write_json(
        config.output_dir / "preparation_summary.json",
        {"column_count": len(manifest), "ontology_class_count": class_count},
    )


def _export(config: CtaRetrievalConfig) -> None:
    summary = export_dart_candidates(
        config.output_dir / "retrieval.pkl",
        config.output_dir / "manifest.jsonl",
        config.output_dir / "ontology.json",
        config.output_dir / "candidates.jsonl",
        config.run_id,
    )
    _write_json(config.output_dir / "export_summary.json", asdict(summary))


def _validate_outputs(config: CtaRetrievalConfig, stage: str) -> None:
    preparation = _read_json(config.output_dir / "preparation_summary.json")
    column_count = int(preparation.get("column_count", 0))
    class_count = int(preparation.get("ontology_class_count", 0))
    if column_count < 1 or class_count < 1:
        raise ValueError("Prepared artifacts are empty")

    summary: dict[str, Any] = {
        "stage": stage,
        "column_count": column_count,
        "ontology_class_count": class_count,
    }
    if stage == "complete":
        candidates = read_candidates_jsonl(config.output_dir / "candidates.jsonl")
        if not candidates:
            raise ValueError("Candidate output is empty")
        expected = min(config.top_k, class_count)
        if len(candidates) != column_count * expected:
            raise ValueError(
                f"Expected {column_count * expected} candidates, got {len(candidates)}"
            )
        summary["candidate_count"] = len(candidates)
    _write_json(config.output_dir / "validation_summary.json", summary)


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())

