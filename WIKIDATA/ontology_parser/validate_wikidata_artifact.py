"""Independent validator for a generated Wikidata ontology artifact."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

from .wikidata_artifact import collect_candidate_data


_QID_PATTERN = re.compile(r"^Q[1-9][0-9]*$")
_REQUIRED_TYPES = {
    "qid": str,
    "label": str,
    "description": str,
    "aliases": list,
    "parents": list,
    "ancestors": list,
    "children": list,
    "examples": list,
}


def _read_ontology(path: Path) -> List[dict]:
    records: List[dict] = []
    with Path(path).open(encoding="utf-8") as handle:
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
                    "Expected object in {0} at line {1}".format(
                        path, line_number
                    )
                )
            records.append(record)
    return records


def validate_artifact(
    train_path: str,
    ontology_path: Path,
    max_examples: int = 10,
) -> Dict[str, int]:
    candidates = collect_candidate_data(train_path, max_examples=max_examples)
    records = _read_ontology(Path(ontology_path))
    violations: List[str] = []
    seen = set()
    qids: List[str] = []

    for line_number, record in enumerate(records, 1):
        for field, expected_type in _REQUIRED_TYPES.items():
            if not isinstance(record.get(field), expected_type):
                violations.append(
                    "line {0}: {1} must be {2}".format(
                        line_number, field, expected_type.__name__
                    )
                )
        qid = record.get("qid")
        if not isinstance(qid, str) or not _QID_PATTERN.fullmatch(qid):
            violations.append("line {0}: invalid qid".format(line_number))
            continue
        if qid in seen:
            violations.append("line {0}: duplicate QID {1}".format(line_number, qid))
        seen.add(qid)
        qids.append(qid)

        parents = record.get("parents")
        if isinstance(parents, list):
            for parent in parents:
                if (
                    not isinstance(parent, dict)
                    or not isinstance(parent.get("qid"), str)
                    or not isinstance(parent.get("label"), str)
                ):
                    violations.append(
                        "line {0}: parents entries require string qid and label".format(
                            line_number
                        )
                    )
                    break
        examples = record.get("examples")
        if isinstance(examples, list) and len(examples) > max_examples:
            violations.append(
                "line {0}: more than {1} examples".format(
                    line_number, max_examples
                )
            )

    if qids != sorted(qids):
        violations.append("ontology QIDs are not sorted")

    for qid in sorted(set(candidates.qids) - seen):
        violations.append("missing training candidate {0}".format(qid))
    for qid in sorted(seen - set(candidates.qids)):
        violations.append("unexpected ontology candidate {0}".format(qid))

    summary = {
        "training_record_count": candidates.record_count,
        "training_candidate_count": len(candidates.qids),
        "ontology_record_count": len(records),
        "violations": len(violations),
    }
    if violations:
        raise ValueError(
            "Ontology validation failed with {0} violation(s):\n{1}".format(
                len(violations),
                "\n".join("- " + violation for violation in violations),
            )
        )
    return summary


def parse_args(argv: List[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a DART Wikidata ontology artifact."
    )
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--ontology", required=True)
    parser.add_argument("--max-examples", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: List[str] = None) -> int:
    args = parse_args(argv)
    summary = validate_artifact(
        args.train_path,
        Path(args.ontology),
        max_examples=args.max_examples,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
