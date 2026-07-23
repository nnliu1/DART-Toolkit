import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from validate_and_build_smoke import (
    assess_training_quality,
    build_contract_ontology,
    inspect_model_zip,
    reconcile_unique_type_metadata,
    select_smoke_records,
    validate_record,
)


def valid_record(qid="Q5"):
    return {
        "anchor_header": "Name",
        "anchor_cells": ["Ada", "Grace"],
        "positive_type_qid": qid,
        "positive_type_label": "human",
        "positive_type_qids": [qid],
        "positive_type_labels": ["human"],
        "positive_type_support": {qid: 1.0},
        "ignored_type_qids": [qid],
        "type_entity_counts": {qid: 2},
        "num_typed_entities": 2,
        "hard_negative_type_qids": ["Q11424"],
        "hard_negative_type_labels": ["film"],
        "table_id": "https://example.test/table",
        "col_index": 0,
        "majority_ratio": 1.0,
    }


class ValidationTests(unittest.TestCase):
    def test_training_quality_does_not_depend_on_unseen_ontology(self):
        result = assess_training_quality(
            invalid_json=0,
            invalid_records=0,
            metadata_total_matches=True,
            metadata_shards_match=True,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["failures"], [])

    def test_valid_record_has_no_errors(self):
        self.assertEqual(validate_record(valid_record()), [])

    def test_rejects_positive_in_negatives_and_length_mismatch(self):
        record = valid_record()
        record["hard_negative_type_qids"] = ["Q5", "Q11424"]
        errors = validate_record(record)
        self.assertIn("positive_in_hard_negatives", errors)
        self.assertIn("hard_negative_label_length_mismatch", errors)

    def test_stratified_smoke_is_deterministic_and_covers_types(self):
        records = [valid_record("Q5"), valid_record("Q5"), valid_record("Q11424")]
        first = select_smoke_records(records, size=2, seed=42)
        second = select_smoke_records(records, size=2, seed=42)
        self.assertEqual(first, second)
        self.assertEqual({r["positive_type_qid"] for r in first}, {"Q5", "Q11424"})

    def test_contract_ontology_contains_positive_and_negative_qids(self):
        ontology = build_contract_ontology([valid_record()])
        self.assertEqual(set(ontology), {"Q5", "Q11424"})
        self.assertEqual(ontology["Q5"]["label"], "human")

    def test_model_zip_requires_core_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.zip"
            with zipfile.ZipFile(path, "w") as archive:
                for name in ("model.safetensors", "config.json", "tokenizer.json"):
                    archive.writestr(f"model/{name}", "x")
            result = inspect_model_zip(path)
        self.assertTrue(result["complete"])
        self.assertEqual(result["missing_required_files"], [])

    def test_reconciles_types_removed_when_labels_are_missing(self):
        result = reconcile_unique_type_metadata(
            metadata_unique_types=7724,
            observed_unique_types=7704,
            label_not_found=20,
        )
        self.assertEqual(result["status"], "explained_by_label_not_found")


if __name__ == "__main__":
    unittest.main()
