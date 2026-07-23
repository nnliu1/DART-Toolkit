"""Schema-v2 JSONL input helpers with legacy compatibility."""
from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Dict, Iterator, List


def resolve_jsonl_paths(path_spec: str) -> List[Path]:
    path = Path(path_spec)
    if path.is_file():
        paths = [path]
    elif path.is_dir():
        paths = sorted(path.glob("train_shard_*.jsonl"))
        if not paths:
            paths = sorted(path.glob("*.jsonl"))
    else:
        paths = [Path(item) for item in sorted(glob.glob(path_spec))]
    paths = [item.resolve() for item in paths if item.is_file()]
    if not paths:
        raise FileNotFoundError(f"No JSONL files resolved from: {path_spec}")
    return paths


def iter_jsonl_records(path_spec: str) -> Iterator[Dict]:
    for path in resolve_jsonl_paths(path_spec):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {path} at line {line_number}: {exc}"
                    ) from exc


def normalize_positive_qids(record: Dict) -> List[str]:
    values = record.get("positive_type_qids") or [record["positive_type_qid"]]
    output = []
    seen = set()
    for qid in values:
        if isinstance(qid, str) and qid and qid not in seen:
            output.append(qid)
            seen.add(qid)
    primary = record.get("positive_type_qid")
    if primary and primary not in seen:
        output.insert(0, primary)
    return output


def normalize_ignored_qids(record: Dict) -> List[str]:
    values = record.get("ignored_type_qids") or []
    return list(dict.fromkeys(qid for qid in values if isinstance(qid, str) and qid))
