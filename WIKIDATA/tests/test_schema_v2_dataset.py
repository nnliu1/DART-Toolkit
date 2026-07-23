import json
import tempfile
import unittest
from pathlib import Path

from WIKIDATA.dart_encoder.schema_v2 import (
    iter_jsonl_records,
    normalize_positive_qids,
    resolve_jsonl_paths,
)
from WIKIDATA.dart_encoder.dataset import CTACollator, CTADataset


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


class SchemaV2DatasetTests(unittest.TestCase):
    def test_collator_flattens_positive_sets_and_preserves_counts(self):
        class Tokenizer:
            def __call__(self, texts, **kwargs):
                return {"texts": list(texts)}

        batch = [
            {
                "query_text": "q1", "pos_text": "p1",
                "pos_texts": ["p1", "p2"], "pos_qids": ["Q1", "Q2"],
                "neg_texts": ["n1"], "neg_qids": ["Q3"],
            },
            {
                "query_text": "q2", "pos_text": "p3",
                "pos_texts": ["p3"], "pos_qids": ["Q4"],
                "neg_texts": [], "neg_qids": [],
            },
        ]
        output = CTACollator(Tokenizer())(batch)
        self.assertEqual(output["pos_counts"], [2, 1])
        self.assertEqual(output["positive_qids_flat"], ["Q1", "Q2", "Q4"])
        self.assertEqual(output["positive_qid_sets"], [["Q1", "Q2"], ["Q4"]])
        self.assertEqual(output["neg_counts"], [1, 0])

    def test_resolves_file_directory_and_glob_in_sorted_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(root / "train_shard_0001.jsonl", [])
            write_jsonl(root / "train_shard_0000.jsonl", [])
            expected = [root / "train_shard_0000.jsonl", root / "train_shard_0001.jsonl"]
            self.assertEqual(resolve_jsonl_paths(str(root)), expected)
            self.assertEqual(resolve_jsonl_paths(str(root / "*.jsonl")), expected)
            self.assertEqual(resolve_jsonl_paths(str(expected[0])), [expected[0]])

    def test_normalizes_legacy_and_multi_positive_records(self):
        self.assertEqual(normalize_positive_qids({"positive_type_qid": "Q5"}), ["Q5"])
        self.assertEqual(
            normalize_positive_qids({
                "positive_type_qid": "Q5",
                "positive_type_qids": ["Q5", "Q215627", "Q5"],
            }),
            ["Q5", "Q215627"],
        )

    def test_iterates_all_shards_with_source_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(root / "train_shard_0000.jsonl", [{"positive_type_qid": "Q1"}])
            write_jsonl(root / "train_shard_0001.jsonl", [{"positive_type_qid": "Q2"}])
            records = list(iter_jsonl_records(str(root)))
            self.assertEqual([r["positive_type_qid"] for r in records], ["Q1", "Q2"])

    def test_dataset_keeps_all_positives_and_removes_ignored_negatives(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ontology = root / "ontology.jsonl"
            write_jsonl(ontology, [
                {"qid": "Q5", "label": "人类"},
                {"qid": "Q215627", "label": "person"},
                {"qid": "Q11424", "label": "film"},
            ])
            write_jsonl(root / "train_shard_0000.jsonl", [{
                "anchor_header": "Name",
                "anchor_cells": ["Ada"],
                "positive_type_qid": "Q5",
                "positive_type_qids": ["Q5", "Q215627"],
                "ignored_type_qids": ["Q215627"],
                "hard_negative_type_qids": ["Q215627", "Q11424"],
                "table_id": "t",
                "col_index": 0,
                "majority_ratio": 1.0,
            }])
            item = CTADataset(str(root), str(ontology))[0]
            self.assertEqual(item["pos_qids"], ["Q5", "Q215627"])
            self.assertEqual(item["neg_qids"], ["Q11424"])
            self.assertIn("人类", item["pos_texts"][0])


if __name__ == "__main__":
    unittest.main()
