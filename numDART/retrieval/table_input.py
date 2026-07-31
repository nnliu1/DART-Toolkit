"""Convert archived real tables into native DART column records."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable
import zipfile

from .models import ColumnManifestRecord


def build_dart_column_inputs(
    archive_path: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    max_cells: int = 10,
) -> list[ColumnManifestRecord]:
    """Builds one native DART input file per real table column.

    Args:
        archive_path: ZIP containing gzipped JSON-lines tables.
        output_dir: Destination directory for per-column JSON files.
        manifest_path: Destination JSONL mapping records to source columns.
        max_cells: Maximum number of non-empty source cells to retain.

    Returns:
        Manifest records in deterministic table-and-column order.

    Raises:
        ValueError: If the sample limit or archive content is invalid.
    """
    if max_cells < 1:
        raise ValueError("max_cells must be at least 1")

    archive = Path(archive_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for stale_file in destination.glob("*.json"):
        stale_file.unlink()

    records: list[ColumnManifestRecord] = []
    with zipfile.ZipFile(archive) as source:
        members = sorted(name for name in source.namelist() if name.endswith(".json.gz"))
        if not members:
            raise ValueError("Table archive contains no .json.gz members")
        for member in members:
            rows = _read_rows(source.read(member), member)
            columns = _column_names(rows, member)
            table_id = member.removesuffix(".json.gz")
            for column_index, column_name in enumerate(columns):
                record = _manifest_record(
                    archive, member, table_id, column_index, column_name
                )
                samples = _sample_column(rows, column_name, max_cells)
                _write_json(destination / f"{record.record_id}.json", {
                    "table_id": record.record_id,
                    "pk_col_header_raw": column_name,
                    "pk_col_header_clean": column_name,
                    "cell_samples_raw": samples,
                    "cell_samples_clean": samples,
                    "gt_uri": "",
                    "gt_ontology": "",
                })
                records.append(record)

    _write_manifest(Path(manifest_path), records)
    return records


def _read_rows(payload: bytes, member: str) -> list[dict[str, Any]]:
    try:
        lines = gzip.decompress(payload).decode("utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid table member {member}: {error}") from error
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Table member {member} must contain JSON object rows")
    return rows


def _column_names(rows: Iterable[dict[str, Any]], member: str) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            name = str(key)
            if name not in seen:
                columns.append(name)
                seen.add(name)
    if not columns:
        raise ValueError(f"Table member {member} contains no columns")
    return columns


def _sample_column(
    rows: Iterable[dict[str, Any]], column_name: str, max_cells: int
) -> list[str]:
    samples: list[str] = []
    for row in rows:
        value = row.get(column_name)
        if value is None:
            continue
        text = _cell_text(value)
        if not text.strip():
            continue
        samples.append(text)
        if len(samples) == max_cells:
            break
    return samples


def _cell_text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _manifest_record(
    archive: Path,
    member: str,
    table_id: str,
    column_index: int,
    column_name: str,
) -> ColumnManifestRecord:
    identity = f"{member}\0{column_index}\0{column_name}".encode("utf-8")
    return ColumnManifestRecord(
        record_id=hashlib.sha256(identity).hexdigest()[:20],
        source_table_id=table_id,
        column_index=column_index,
        column_name=column_name,
        source_archive=str(archive.resolve()),
        source_member=member,
    )


def _write_manifest(path: Path, records: Iterable[ColumnManifestRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True))
            output.write("\n")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
