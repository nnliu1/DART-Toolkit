"""Create a deterministic page-disjoint train/validation JSONL split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Dict, IO, List, Optional, Set, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def resolve_input_shards(input_dir: Path) -> List[Path]:
    """Return the sorted source training shards.

    Args:
        input_dir: Directory containing `train_shard_*.jsonl` files.

    Returns:
        Resolved shard paths in filename order.

    Raises:
        FileNotFoundError: If no source training shards are present.
    """

    paths = sorted(
        path.resolve()
        for path in Path(input_dir).glob("train_shard_*.jsonl")
        if path.is_file()
    )
    if not paths:
        raise FileNotFoundError(
            "No train_shard_*.jsonl files found in: {0}".format(input_dir)
        )
    return paths


def page_key(table_id: object) -> Tuple[str, bool]:
    """Convert a table identifier into its Wikipedia page group.

    Args:
        table_id: Value from a training record's `table_id` field.

    Returns:
        A pair containing the group key and whether the identifier was
        malformed. Malformed identifiers retain their complete string value.
    """

    value = table_id if isinstance(table_id, str) else str(table_id or "")
    parsed = urlsplit(value)
    valid = (
        parsed.scheme in {"http", "https"}
        and parsed.netloc.lower() == "en.wikipedia.org"
        and parsed.path.startswith("/wiki/")
    )
    if not valid:
        return value, True

    query = [
        (name, item)
        for name, item in parse_qsl(parsed.query, keep_blank_values=True)
        if name != "table_no"
    ]
    return (
        urlunsplit(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                parsed.path,
                urlencode(query),
                "",
            )
        ),
        False,
    )


def is_validation_group(
    group_key: str,
    val_ratio: float,
    seed: int,
) -> bool:
    """Return whether a page group belongs to validation.

    Args:
        group_key: Stable page identifier.
        val_ratio: Desired validation fraction strictly between zero and one.
        seed: Integer used to create a reproducible alternative assignment.

    Returns:
        True when the group is assigned to validation.

    Raises:
        ValueError: If `val_ratio` is outside the open interval `(0, 1)`.
    """

    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be strictly between 0 and 1")

    ratio = Fraction(str(val_ratio)).limit_denominator(1_000_000)
    payload = "{0}:{1}".format(seed, group_key).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % ratio.denominator < ratio.numerator


class _ShardWriter:
    """Write JSONL records to bounded, consecutively numbered shards."""

    def __init__(self, output_dir: Path, shard_size: int) -> None:
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.record_count = 0
        self.shard_count = 0
        self._records_in_shard = 0
        self._handle: Optional[IO[str]] = None

    def write(self, line: str) -> None:
        """Write one complete JSON line."""

        if self._handle is None or self._records_in_shard >= self.shard_size:
            self._open_next_shard()
        assert self._handle is not None
        self._handle.write(line.rstrip("\r\n"))
        self._handle.write("\n")
        self._records_in_shard += 1
        self.record_count += 1

    def close(self) -> None:
        """Flush, synchronize, and close the active shard."""

        if self._handle is None:
            return
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        self._handle = None

    def _open_next_shard(self) -> None:
        self.close()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "train_shard_{0:04d}.jsonl".format(
            self.shard_count
        )
        self._handle = path.open("w", encoding="utf-8", newline="\n")
        self.shard_count += 1
        self._records_in_shard = 0


def _positive_qids(record: Dict[str, object]) -> Set[str]:
    values = record.get("positive_type_qids")
    output = {
        value
        for value in values
        if isinstance(value, str) and value
    } if isinstance(values, list) else set()
    primary = record.get("positive_type_qid")
    if isinstance(primary, str) and primary:
        output.add(primary)
    return output


def _write_metadata(path: Path, metadata: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _install_output(
    temporary_dir: Path,
    output_dir: Path,
    overwrite: bool,
) -> None:
    backup_dir: Optional[Path] = None
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                "Output already exists; pass --overwrite to replace it: "
                "{0}".format(output_dir)
            )
        backup_dir = output_dir.with_name(
            ".{0}.backup-{1}".format(output_dir.name, uuid.uuid4().hex)
        )
        output_dir.rename(backup_dir)

    try:
        os.replace(str(temporary_dir), str(output_dir))
    except BaseException:
        if backup_dir is not None and backup_dir.exists():
            backup_dir.rename(output_dir)
        raise
    if backup_dir is not None:
        shutil.rmtree(str(backup_dir))


def split_dataset(
    input_dir: Path,
    output_dir: Path,
    val_ratio: float = 0.1,
    seed: int = 42,
    shard_size: int = 50_000,
    overwrite: bool = False,
) -> Dict[str, object]:
    """Split CTA records into deterministic page-disjoint partitions.

    Args:
        input_dir: Directory containing the source training shards.
        output_dir: New directory that will contain train, validation, and
            metadata outputs.
        val_ratio: Fraction of page groups assigned to validation.
        seed: Stable assignment seed.
        shard_size: Maximum records written to one output JSONL shard.
        overwrite: Whether an existing final output may be replaced.

    Returns:
        Metadata describing the generated partitions and validation checks.

    Raises:
        FileExistsError: If the output exists and `overwrite` is false.
        ValueError: If configuration, JSON input, or split validation fails.
    """

    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be strictly between 0 and 1")
    if shard_size <= 0:
        raise ValueError("shard_size must be greater than zero")

    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            "Output already exists; pass --overwrite to replace it: "
            "{0}".format(output_dir)
        )
    source_paths = resolve_input_shards(input_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=".{0}.tmp-".format(output_dir.name),
            dir=str(output_dir.parent),
        )
    )

    train_writer = _ShardWriter(temporary_dir / "train", shard_size)
    validation_writer = _ShardWriter(temporary_dir / "val", shard_size)
    source_record_count = 0
    malformed_table_id_count = 0
    train_pages: Set[str] = set()
    validation_pages: Set[str] = set()
    train_tables: Set[str] = set()
    validation_tables: Set[str] = set()
    train_primary_types: Counter = Counter()
    validation_primary_types: Counter = Counter()
    train_positive_types: Set[str] = set()
    validation_positive_types: Set[str] = set()

    try:
        for source_path in source_paths:
            with source_path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            "Invalid JSON in {0} at line {1}: {2}".format(
                                source_path, line_number, exc
                            )
                        ) from exc
                    if not isinstance(record, dict):
                        raise ValueError(
                            "Expected JSON object in {0} at line {1}".format(
                                source_path, line_number
                            )
                        )

                    source_record_count += 1
                    table_id = record.get("table_id")
                    group_key, malformed = page_key(table_id)
                    malformed_table_id_count += int(malformed)
                    validation = is_validation_group(
                        group_key, val_ratio, seed
                    )
                    writer = validation_writer if validation else train_writer
                    writer.write(line)

                    pages = validation_pages if validation else train_pages
                    tables = validation_tables if validation else train_tables
                    primary_types = (
                        validation_primary_types
                        if validation
                        else train_primary_types
                    )
                    positive_types = (
                        validation_positive_types
                        if validation
                        else train_positive_types
                    )
                    pages.add(group_key)
                    tables.add(
                        table_id
                        if isinstance(table_id, str)
                        else str(table_id or "")
                    )
                    primary = record.get("positive_type_qid")
                    if isinstance(primary, str) and primary:
                        primary_types[primary] += 1
                    positive_types.update(_positive_qids(record))

        train_writer.close()
        validation_writer.close()

        if train_writer.record_count == 0 or validation_writer.record_count == 0:
            raise ValueError("Both train and validation partitions must be non-empty")
        if (
            train_writer.record_count + validation_writer.record_count
            != source_record_count
        ):
            raise ValueError("Output record counts do not match source records")

        page_intersection = train_pages & validation_pages
        table_intersection = train_tables & validation_tables
        if page_intersection:
            raise ValueError("Train and validation page groups overlap")
        if table_intersection:
            raise ValueError("Train and validation table IDs overlap")

        validation_seen = validation_positive_types & train_positive_types
        validation_unseen = validation_positive_types - train_positive_types
        metadata: Dict[str, object] = {
            "schema_version": 1,
            "assignment": "sha256(seed:page_key) modulo ratio denominator",
            "seed": seed,
            "validation_ratio_requested": val_ratio,
            "validation_ratio_actual": (
                validation_writer.record_count / source_record_count
            ),
            "shard_size": shard_size,
            "source_shards": [path.name for path in source_paths],
            "source_record_count": source_record_count,
            "train_record_count": train_writer.record_count,
            "validation_record_count": validation_writer.record_count,
            "train_shard_count": train_writer.shard_count,
            "validation_shard_count": validation_writer.shard_count,
            "train_page_count": len(train_pages),
            "validation_page_count": len(validation_pages),
            "train_table_count": len(train_tables),
            "validation_table_count": len(validation_tables),
            "page_intersection_count": len(page_intersection),
            "table_intersection_count": len(table_intersection),
            "train_primary_type_count": len(train_primary_types),
            "validation_primary_type_count": len(validation_primary_types),
            "train_positive_type_count": len(train_positive_types),
            "validation_positive_type_count": len(validation_positive_types),
            "validation_seen_positive_type_count": len(validation_seen),
            "validation_unseen_positive_type_count": len(validation_unseen),
            "validation_unseen_positive_types": sorted(validation_unseen),
            "malformed_table_id_count": malformed_table_id_count,
        }
        _write_metadata(temporary_dir / "split_metadata.json", metadata)
        _install_output(temporary_dir, output_dir, overwrite)
        return metadata
    except BaseException:
        train_writer.close()
        validation_writer.close()
        if temporary_dir.exists():
            shutil.rmtree(str(temporary_dir))
        raise


def parse_args(
    argv: Optional[List[str]] = None,
) -> argparse.Namespace:
    """Parse standalone splitter arguments.

    Args:
        argv: Optional explicit argument list. `None` reads process arguments.

    Returns:
        Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic page-disjoint train/validation split "
            "from CTA JSONL shards."
        )
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-size", type=int, default=50_000)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output only after the new split validates.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Run the standalone data splitter.

    Args:
        argv: Optional explicit argument list for tests and embedding.

    Returns:
        Process exit code zero after a successful split.
    """

    args = parse_args(argv)
    metadata = split_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        shard_size=args.shard_size,
        overwrite=args.overwrite,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
