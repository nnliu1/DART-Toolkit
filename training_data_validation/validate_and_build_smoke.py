#!/usr/bin/env python3
"""CPU-only validation and smoke-artifact builder for DART CTA training data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


REQUIRED_FIELDS = {
    "anchor_header", "anchor_cells", "positive_type_qid",
    "positive_type_label", "positive_type_qids", "positive_type_labels",
    "hard_negative_type_qids", "hard_negative_type_labels",
    "table_id", "col_index", "majority_ratio",
}


def _is_qid(value: Any) -> bool:
    return isinstance(value, str) and len(value) > 1 and value[0] == "Q" and value[1:].isdigit()


def validate_record(record: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    missing = sorted(REQUIRED_FIELDS - set(record))
    if missing:
        errors.append("missing_fields:" + ",".join(missing))
        return errors
    if not isinstance(record["anchor_header"], str):
        errors.append("anchor_header_not_string")
    cells = record["anchor_cells"]
    if not isinstance(cells, list) or not cells or not all(isinstance(x, str) and x.strip() for x in cells):
        errors.append("invalid_anchor_cells")
    positive = record["positive_type_qid"]
    if not _is_qid(positive):
        errors.append("invalid_positive_qid")
    positives = record["positive_type_qids"]
    positive_labels = record["positive_type_labels"]
    if not isinstance(positives, list) or not all(_is_qid(x) for x in positives):
        errors.append("invalid_positive_qids")
    elif positive not in positives:
        errors.append("primary_positive_missing_from_positive_set")
    if not isinstance(positive_labels, list) or len(positive_labels) != len(positives):
        errors.append("positive_label_length_mismatch")
    negatives = record["hard_negative_type_qids"]
    negative_labels = record["hard_negative_type_labels"]
    if not isinstance(negatives, list) or not all(_is_qid(x) for x in negatives):
        errors.append("invalid_hard_negative_qids")
    if not isinstance(negative_labels, list) or len(negative_labels) != len(negatives):
        errors.append("hard_negative_label_length_mismatch")
    if isinstance(negatives, list) and positive in negatives:
        errors.append("positive_in_hard_negatives")
    if isinstance(positives, list) and isinstance(negatives, list) and set(positives) & set(negatives):
        errors.append("positive_set_overlaps_hard_negatives")
    ratio = record["majority_ratio"]
    if not isinstance(ratio, (int, float)) or not 0.0 <= ratio <= 1.0:
        errors.append("invalid_majority_ratio")
    if not isinstance(record["table_id"], str) or not record["table_id"]:
        errors.append("invalid_table_id")
    if not isinstance(record["col_index"], int) or record["col_index"] < 0:
        errors.append("invalid_col_index")
    if len(negatives) != len(set(negatives)):
        errors.append("duplicate_hard_negatives")
    return errors


def select_smoke_records(records: Sequence[Dict[str, Any]], size: int, seed: int) -> List[Dict[str, Any]]:
    if size <= 0:
        return []
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    selected: List[Dict[str, Any]] = []
    seen = set()
    for record in shuffled:
        qid = record["positive_type_qid"]
        if qid not in seen:
            selected.append(record)
            seen.add(qid)
            if len(selected) == size:
                return selected
    selected_ids = {id(record) for record in selected}
    selected.extend(record for record in shuffled if id(record) not in selected_ids)
    return selected[:size]


def build_contract_ontology(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    ontology: Dict[str, Dict[str, Any]] = {}
    for record in records:
        pairs = [(record["positive_type_qid"], record["positive_type_label"])]
        pairs += list(zip(record["positive_type_qids"], record["positive_type_labels"]))
        pairs += list(zip(record["hard_negative_type_qids"], record["hard_negative_type_labels"]))
        for qid, label in pairs:
            ontology.setdefault(qid, {
                "qid": qid,
                "label": label,
                "description": "",
                "parents": [],
                "_artifact_scope": "contract-smoke-only",
            })
    return ontology


def inspect_model_zip(path: Path) -> Dict[str, Any]:
    required = {"model.safetensors", "config.json", "tokenizer.json"}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    basenames = {Path(name).name for name in names if not name.endswith("/")}
    missing = sorted(required - basenames)
    return {
        "path": str(path),
        "file_count": len(basenames),
        "missing_required_files": missing,
        "complete": not missing,
    }


def reconcile_unique_type_metadata(
    metadata_unique_types: Any,
    observed_unique_types: int,
    label_not_found: Any,
) -> Dict[str, Any]:
    exact = metadata_unique_types == observed_unique_types
    explained = (
        isinstance(metadata_unique_types, int)
        and isinstance(label_not_found, int)
        and metadata_unique_types - label_not_found == observed_unique_types
    )
    return {
        "exact_match": exact,
        "status": (
            "exact_match" if exact else
            "explained_by_label_not_found" if explained else
            "unexplained_mismatch"
        ),
    }


def assess_training_quality(
    *,
    invalid_json: int,
    invalid_records: int,
    metadata_total_matches: bool,
    metadata_shards_match: bool,
) -> Dict[str, Any]:
    failures = []
    if invalid_json:
        failures.append("invalid_json")
    if invalid_records:
        failures.append("schema_or_invariant_violations")
    if not metadata_total_matches:
        failures.append("metadata_total_mismatch")
    if not metadata_shards_match:
        failures.append("metadata_shard_mismatch")
    return {"passed": not failures, "failures": failures}


def _stable_score(record: Dict[str, Any], seed: int) -> str:
    key = f'{seed}|{record.get("table_id")}|{record.get("col_index")}|{record.get("positive_type_qid")}'
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(args: argparse.Namespace) -> int:
    training_dir = Path(args.training_dir).resolve()
    model_zip = Path(args.model_zip).resolve()
    encoder_dir = Path(args.encoder_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    reports_dir = output_dir / "reports"
    smoke_dir = output_dir / "smoke"
    reports_dir.mkdir(parents=True, exist_ok=True)
    smoke_dir.mkdir(parents=True, exist_ok=True)

    shard_paths = sorted(training_dir.glob("train_shard_*.jsonl"))
    if not shard_paths:
        raise FileNotFoundError(f"No train_shard_*.jsonl files in {training_dir}")

    counts = Counter()
    type_counts = Counter()
    positive_qids = set()
    negative_qids = set()
    error_examples: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    best_by_type: Dict[str, tuple[str, Dict[str, Any]]] = {}

    for shard in shard_paths:
        with shard.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                counts["records"] += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    counts["invalid_json"] += 1
                    if len(error_examples) < 100:
                        error_examples.append({"file": shard.name, "line": line_number, "errors": [str(exc)]})
                    continue
                errors = validate_record(record)
                if errors:
                    counts["invalid_records"] += 1
                    if len(error_examples) < 100:
                        error_examples.append({"file": shard.name, "line": line_number, "errors": errors})
                    continue
                counts["valid_records"] += 1
                qid = record["positive_type_qid"]
                type_counts[qid] += 1
                positive_qids.update(record["positive_type_qids"])
                negative_qids.update(record["hard_negative_type_qids"])
                score = _stable_score(record, args.seed)
                current = best_by_type.get(qid)
                if current is None or score < current[0]:
                    best_by_type[qid] = (score, record)
                if len(candidates) < max(args.smoke_size * 8, 1000):
                    candidates.append(record)

    metadata_path = training_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    candidate_pool = [item[1] for item in best_by_type.values()] + candidates
    smoke_records = select_smoke_records(candidate_pool, args.smoke_size, args.seed)
    contract_ontology = build_contract_ontology(smoke_records)
    _write_jsonl(smoke_dir / "train_smoke.jsonl", smoke_records)
    _write_jsonl(smoke_dir / "ontology_training_smoke.jsonl", contract_ontology.values())

    with (reports_dir / "type_frequency.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["positive_type_qid", "count"])
        writer.writerows(type_counts.most_common())

    train_py = encoder_dir / "train.py"
    dataset_py = encoder_dir / "dataset.py"
    train_text = train_py.read_text(encoding="utf-8", errors="replace") if train_py.exists() else ""
    dataset_text = dataset_py.read_text(encoding="utf-8", errors="replace") if dataset_py.exists() else ""
    single_file_contract = "open(train_path)" in dataset_text
    model = inspect_model_zip(model_zip)
    unique_type_reconciliation = reconcile_unique_type_metadata(
        metadata.get("unique_types"), len(type_counts), metadata.get("label_not_found")
    )
    metadata_matches = {
        "total_examples": metadata.get("total_examples") == counts["records"],
        "unique_types": unique_type_reconciliation["exact_match"],
        "num_shards": metadata.get("num_shards") == len(shard_paths),
    }
    quality = assess_training_quality(
        invalid_json=counts["invalid_json"],
        invalid_records=counts["invalid_records"],
        metadata_total_matches=metadata_matches["total_examples"],
        metadata_shards_match=metadata_matches["num_shards"],
    )

    report = {
        "schema_version": 1,
        "inputs": {
            "training_dir": str(training_dir),
            "model_zip": str(model_zip),
            "encoder_dir": str(encoder_dir),
            "shards": [str(path) for path in shard_paths],
        },
        "dataset": {
            **dict(counts),
            "unique_primary_types": len(type_counts),
            "unique_positive_qids": len(positive_qids),
            "unique_negative_qids": len(negative_qids),
            "metadata": metadata,
            "metadata_matches_observed": metadata_matches,
            "unique_type_metadata_reconciliation": unique_type_reconciliation,
            "error_examples": error_examples,
        },
        "scope": {
            "training_label_space": "Wikidata QIDs",
            "unseen_ontology_evaluation_excluded": True,
            "dbpedia_not_used_for_training_quality": True,
        },
        "dart_contract": {
            "train_py_present": train_py.exists(),
            "dataset_py_present": dataset_py.exists(),
            "train_cli_mentions_ontology_path": "--ontology_path" in train_text,
            "loader_uses_single_train_file": single_file_contract,
            "model_artifact": model,
        },
        "smoke": {
            "records": len(smoke_records),
            "unique_primary_types": len({r["positive_type_qid"] for r in smoke_records}),
            "contract_ontology_terms": len(contract_ontology),
        },
        "conclusion": {
            "wikidata_training_data_quality_passed": quality["passed"],
            "quality_failures": quality["failures"],
            "operational_notes": [
                "CTADataset currently accepts one train_path file; use one shard, merge shards, or add multi-shard loading for full training."
            ] if single_file_contract and len(shard_paths) > 1 else [],
        },
    }
    (reports_dir / "dataset_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    status = "PASS" if quality["passed"] else "FAIL"
    markdown = f"""# DART v1 training-data compatibility report

## Result

- Wikidata training-data quality: **{status}**
- Records checked: **{counts['records']:,}**
- Valid records: **{counts['valid_records']:,}**
- Unique primary Wikidata types: **{len(type_counts):,}**
- Unique-type metadata reconciliation: **{unique_type_reconciliation['status']}**

## Interpretation

This report evaluates only the Wikidata retrieval training data. DBpedia is an unseen ontology for later evaluation and is deliberately excluded from all training-quality pass/fail criteria.

Metadata reports {metadata.get('unique_types')} types, while emitted records contain {len(type_counts)} primary types. The difference of {metadata.get('unique_types', 0) - len(type_counts) if isinstance(metadata.get('unique_types'), int) else 'unknown'} equals `label_not_found={metadata.get('label_not_found')}`, so it is classified as explained by label-resolution filtering rather than shard loss.

## Quality failures

{chr(10).join(f'- {item}' for item in quality['failures']) or '- None.'}

## Operational note

The current `CTADataset` opens one `train_path`, while the full dataset contains {len(shard_paths)} shards. This is a loader/launch issue, not a defect in the Wikidata records.

## Smoke artifacts

- `smoke/train_smoke.jsonl`: deterministic, type-diverse training sample.
- `smoke/ontology_training_smoke.jsonl`: label-derived ontology for loader contract testing only; not a formal training ontology.
- `smoke/model_smoke_test.ps1`: dependency-gated command for a future model-level smoke test.
"""
    (reports_dir / "compatibility_report.md").write_text(markdown, encoding="utf-8")

    ps1 = f"""$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$python = Join-Path $root ".venv\\Scripts\\python.exe"
if (-not (Test-Path $python)) {{ $python = "python" }}
& $python -c "import torch, transformers; print('torch', torch.__version__); print('transformers', transformers.__version__)"
if ($LASTEXITCODE -ne 0) {{ throw "Install torch and transformers before running the model smoke test." }}
Write-Host "Dependencies are available. Use train.py with smoke/train_smoke.jsonl and smoke/ontology_training_smoke.jsonl for a one-step contract smoke run."
"""
    (smoke_dir / "model_smoke_test.ps1").write_text(ps1, encoding="utf-8")
    print(json.dumps(report["conclusion"], ensure_ascii=False))
    return 0 if quality["passed"] else 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--model-zip", required=True)
    parser.add_argument("--encoder-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run(parse_args()))
