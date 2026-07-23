"""Pure helpers for constructing a Wikidata ontology training artifact."""
from __future__ import annotations

import glob
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List


_QID_PATTERN = re.compile(r"^Q[1-9][0-9]*$")


@dataclass
class CandidateData:
    """Candidate type universe and positive column examples."""

    qids: List[str]
    examples: Dict[str, List[str]]
    record_count: int


def _resolve_training_paths(path_spec: str) -> List[Path]:
    path = Path(path_spec)
    if path.is_file():
        paths = [path]
    elif path.is_dir():
        paths = sorted(path.glob("train_shard_*.jsonl"))
    else:
        paths = [Path(item) for item in sorted(glob.glob(path_spec))]

    resolved = [item.resolve() for item in paths if item.is_file()]
    if not resolved:
        raise FileNotFoundError(
            "No training JSONL shards resolved from: {0}".format(path_spec)
        )
    return resolved


def iter_training_records(path_spec: str) -> Iterator[dict]:
    """Yield JSON objects from a training file, directory, or glob."""

    for path in _resolve_training_paths(path_spec):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "Invalid JSON in {0} at line {1}: {2}".format(
                            path, line_number, exc
                        )
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        "Expected a JSON object in {0} at line {1}".format(
                            path, line_number
                        )
                    )
                yield record


def _valid_qids(values: object) -> List[str]:
    if not isinstance(values, list):
        return []
    return [
        value
        for value in values
        if isinstance(value, str) and _QID_PATTERN.fullmatch(value)
    ]


def _positive_qids(record: dict) -> List[str]:
    values = _valid_qids(record.get("positive_type_qids"))
    primary = record.get("positive_type_qid")
    if (
        isinstance(primary, str)
        and _QID_PATTERN.fullmatch(primary)
        and primary not in values
    ):
        values.insert(0, primary)
    return list(dict.fromkeys(values))


def collect_candidate_data_from_records(
    records: Iterable[dict],
    max_examples: int = 10,
) -> CandidateData:
    """Collect the ontology candidate union and positive-only cell examples."""

    if max_examples <= 0:
        raise ValueError("max_examples must be greater than zero")

    candidate_qids = set()
    examples: Dict[str, List[str]] = {}
    seen_examples: Dict[str, set] = {}
    record_count = 0

    for record in records:
        record_count += 1
        positives = _positive_qids(record)
        negatives = _valid_qids(record.get("hard_negative_type_qids"))
        candidate_qids.update(positives)
        candidate_qids.update(negatives)

        cells = record.get("anchor_cells")
        if not isinstance(cells, list):
            continue

        for qid in positives:
            qid_examples = examples.setdefault(qid, [])
            qid_seen = seen_examples.setdefault(qid, set())
            for cell in cells:
                if not isinstance(cell, str):
                    continue
                value = cell.strip()
                if not value or value in qid_seen:
                    continue
                qid_seen.add(value)
                if len(qid_examples) < max_examples:
                    qid_examples.append(value)

    return CandidateData(
        qids=sorted(candidate_qids),
        examples=examples,
        record_count=record_count,
    )


def collect_candidate_data(
    path_spec: str,
    max_examples: int = 10,
) -> CandidateData:
    """Collect candidate data directly from resolved training shards."""

    return collect_candidate_data_from_records(
        iter_training_records(path_spec),
        max_examples=max_examples,
    )
