"""HPC CLI for building the offline Wikidata ontology artifact."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

from .wikidata_artifact import (
    BuildResult,
    build_artifact,
    collect_candidate_data,
)


def parse_args(argv: List[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build DART's Wikidata ontology artifact from local training "
            "shards and kgdata RocksDB databases."
        )
    )
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--wd-db", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--max-ancestor-depth", type=int, default=32)
    return parser.parse_args(argv)


def expected_database_paths(wd_db: Path) -> Dict[str, Path]:
    root = Path(wd_db)
    return {
        "entity": root / "entities.db",
        "label": root / "entity_labels.db",
        "class": root / "classes.db",
    }


def open_kgdata_databases(wd_db: Path) -> Tuple[object, object, object]:
    """Open each RocksDB subdirectory read-only using kgdata 5.x APIs."""

    paths = expected_database_paths(wd_db)
    missing = [str(path) for path in paths.values() if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "Required kgdata database directories are missing: {0}".format(
                ", ".join(missing)
            )
        )

    from kgdata.wikidata.db import (  # type: ignore
        get_entity_db,
        get_entity_label_db,
    )

    try:
        from kgdata.wikidata.db import get_class_db  # type: ignore
    except ImportError:
        from kgdata.wikidata.db import (  # type: ignore
            get_wdclass_db as get_class_db,
        )

    entity_db = get_entity_db(str(paths["entity"]), read_only=True)
    label_db = get_entity_label_db(str(paths["label"]), read_only=True)
    class_db = get_class_db(str(paths["class"]), read_only=True)
    return entity_db, label_db, class_db


def artifact_paths(output: Path) -> Dict[str, Path]:
    output = Path(output)
    base = output.name[:-6] if output.name.endswith(".jsonl") else output.name
    return {
        "ontology": output,
        "metadata": output.with_name(base + ".meta.json"),
        "missing": output.with_name(base + ".missing.jsonl"),
    }


def _atomic_write_lines(path: Path, records: List[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=str(path.parent),
            prefix="." + path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            for record in records:
                handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=False)
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        if temporary_name:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
        raise


def _atomic_write_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=str(path.parent),
            prefix="." + path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        if temporary_name:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
        raise


def write_artifacts(
    output: Path,
    result: BuildResult,
    metadata: Mapping,
) -> Dict[str, Path]:
    paths = artifact_paths(output)
    merged_metadata = dict(metadata)
    merged_metadata.update(result.stats)
    _atomic_write_lines(paths["ontology"], result.records)
    _atomic_write_json(paths["metadata"], merged_metadata)
    _atomic_write_lines(paths["missing"], result.missing)
    return paths


def main(argv: List[str] = None) -> int:
    args = parse_args(argv)
    candidates = collect_candidate_data(
        args.train_path,
        max_examples=args.max_examples,
    )
    entity_db, label_db, class_db = open_kgdata_databases(Path(args.wd_db))
    result = build_artifact(
        candidates,
        entity_db,
        label_db,
        class_db,
        max_ancestor_depth=args.max_ancestor_depth,
    )
    paths = write_artifacts(
        Path(args.output),
        result,
        {
            "schema_version": 1,
            "train_path": str(Path(args.train_path).resolve()),
            "wd_db": str(Path(args.wd_db).resolve()),
            "max_examples": args.max_examples,
            "max_ancestor_depth": args.max_ancestor_depth,
            "training_record_count": candidates.record_count,
        },
    )
    print(
        json.dumps(
            {name: str(path) for name, path in paths.items()},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
