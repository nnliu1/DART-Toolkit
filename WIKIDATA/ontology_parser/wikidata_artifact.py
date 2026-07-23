"""Pure helpers for constructing a Wikidata ontology training artifact."""
from __future__ import annotations

import glob
import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence


_QID_PATTERN = re.compile(r"^Q[1-9][0-9]*$")


@dataclass
class CandidateData:
    """Candidate type universe and positive column examples."""

    qids: List[str]
    examples: Dict[str, List[str]]
    record_count: int


@dataclass
class BuildResult:
    """Serialized records plus diagnostics from ontology construction."""

    records: List[dict]
    missing: List[dict]
    stats: Dict[str, int]


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


def _db_get(database: object, qid: str) -> Optional[object]:
    try:
        return database[qid]  # type: ignore[index]
    except (KeyError, TypeError):
        return None


def _field(value: object, name: str) -> object:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _english_value(value: object) -> Optional[str]:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, dict):
        if "en" in value:
            return _english_value(value["en"])
        if value.get("language") == "en":
            return _english_value(value.get("value"))
        for key in ("value", "text", "label"):
            if key in value:
                text = _english_value(value[key])
                if text:
                    return text
        return None
    language = getattr(value, "language", None)
    if language is None:
        language = getattr(value, "lang", None)
    if language not in (None, "en"):
        return None
    lang2value = getattr(value, "lang2value", None)
    if isinstance(lang2value, dict):
        return _english_value(lang2value.get("en"))
    for name in ("value", "text", "label"):
        nested = getattr(value, name, None)
        if nested is not None and nested is not value:
            text = _english_value(nested)
            if text:
                return text
    return None


def _english_aliases(value: object) -> List[str]:
    if isinstance(value, dict):
        if "en" in value:
            return _english_aliases(value["en"])
        if value.get("language") == "en":
            return _english_aliases(value.get("value"))
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Sequence):
        aliases: List[str] = []
        for item in value:
            text = _english_value(item)
            if text:
                aliases.append(text)
        return aliases
    lang2value = getattr(value, "lang2value", None)
    if isinstance(lang2value, dict):
        return _english_aliases(lang2value.get("en"))
    nested = getattr(value, "values", None)
    if nested is not None:
        return _english_aliases(nested)
    return []


def _normalize_aliases(value: object, primary_label: str) -> List[str]:
    output: List[str] = []
    seen = {primary_label.casefold()}
    for alias in _english_aliases(value):
        key = alias.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(alias)
    return output


def _direct_parents(class_value: object) -> List[str]:
    raw = _field(class_value, "parents")
    if not isinstance(raw, Sequence) or isinstance(raw, str):
        return []
    return sorted(
        set(
            parent
            for parent in raw
            if isinstance(parent, str) and _QID_PATTERN.fullmatch(parent)
        )
    )


def build_artifact(
    candidate_data: CandidateData,
    entity_db: object,
    label_db: object,
    class_db: object,
    max_ancestor_depth: int = 32,
) -> BuildResult:
    """Build deterministic ontology records from database-like mappings."""

    if max_ancestor_depth <= 0:
        raise ValueError("max_ancestor_depth must be greater than zero")

    records: List[dict] = []
    missing_records: List[dict] = []
    candidate_set = set(candidate_data.qids)
    parent_cache: Dict[str, List[str]] = {}

    def parents_for(qid: str) -> List[str]:
        if qid not in parent_cache:
            parent_cache[qid] = _direct_parents(_db_get(class_db, qid))
        return parent_cache[qid]

    def label_for(qid: str) -> str:
        label = _english_value(_db_get(label_db, qid))
        if label:
            return label
        entity = _db_get(entity_db, qid)
        if entity is not None:
            label = _english_value(_field(entity, "label"))
        return label or qid

    children_by_parent: Dict[str, List[str]] = defaultdict(list)
    for child_qid in sorted(candidate_set):
        for parent_qid in parents_for(child_qid):
            if parent_qid in candidate_set:
                children_by_parent[parent_qid].append(child_qid)

    for qid in sorted(candidate_data.qids):
        missing: List[str] = []
        entity = _db_get(entity_db, qid)
        label_entry = _db_get(label_db, qid)
        class_entry = _db_get(class_db, qid)

        if entity is None:
            missing.append("entity")
        label = label_for(qid)
        if not label:
            label = qid
        if label == qid:
            missing.append("label")

        description = ""
        aliases: List[str] = []
        if entity is not None:
            description = (
                _english_value(_field(entity, "description")) or ""
            )
            aliases = _normalize_aliases(_field(entity, "aliases"), label)
        if not description:
            missing.append("description")
        if not aliases:
            missing.append("aliases")
        if class_entry is None:
            missing.append("class")

        direct_parent_qids = parents_for(qid)
        parents = [
            {
                "qid": parent_qid,
                "label": label_for(parent_qid),
            }
            for parent_qid in direct_parent_qids
        ]

        ancestors: List[str] = []
        seen_ancestors = {qid}
        queue = deque((parent_qid, 1) for parent_qid in direct_parent_qids)
        while queue:
            ancestor_qid, depth = queue.popleft()
            if ancestor_qid in seen_ancestors:
                continue
            seen_ancestors.add(ancestor_qid)
            ancestors.append(ancestor_qid)
            if depth >= max_ancestor_depth:
                continue
            for parent_qid in parents_for(ancestor_qid):
                if parent_qid not in seen_ancestors:
                    queue.append((parent_qid, depth + 1))

        records.append(
            {
                "qid": qid,
                "label": label,
                "description": description,
                "aliases": aliases,
                "parents": parents,
                "ancestors": ancestors,
                "children": sorted(children_by_parent.get(qid, [])),
                "examples": list(candidate_data.examples.get(qid, [])),
            }
        )
        if missing:
            missing_records.append({"qid": qid, "missing": missing})

    stats = {
        "candidate_count": len(candidate_data.qids),
        "record_count": len(records),
        "missing_record_count": len(missing_records),
        "with_description_count": sum(
            1 for record in records if record["description"]
        ),
        "with_aliases_count": sum(1 for record in records if record["aliases"]),
        "with_parents_count": sum(1 for record in records if record["parents"]),
        "with_ancestors_count": sum(
            1 for record in records if record["ancestors"]
        ),
        "with_children_count": sum(
            1 for record in records if record["children"]
        ),
        "with_examples_count": sum(
            1 for record in records if record["examples"]
        ),
    }
    return BuildResult(records=records, missing=missing_records, stats=stats)
