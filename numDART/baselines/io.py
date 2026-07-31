from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .contracts import CandidateRecord


def write_candidates_jsonl(
    path: str | Path, records: Iterable[CandidateRecord]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def read_candidates_jsonl(path: str | Path) -> list[CandidateRecord]:
    records: list[CandidateRecord] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(CandidateRecord.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid candidate record at line {line_number}: {error}"
                ) from error
    return records

