import json
import tempfile
import unittest
from pathlib import Path

from WIKIDATA.ontology_parser.wikidata_artifact import (
    collect_candidate_data,
    collect_candidate_data_from_records,
    iter_training_records,
)


class CandidateExtractionTests(unittest.TestCase):
    def test_collects_positive_legacy_and_negative_qids_in_sorted_order(self):
        records = [
            {
                "positive_type_qid": "Q2",
                "positive_type_qids": ["Q2", "Q1", "invalid"],
                "hard_negative_type_qids": ["Q3", "Q2", ""],
                "anchor_cells": ["alpha", "beta"],
            },
            {
                "positive_type_qid": "Q4",
                "hard_negative_type_qids": [],
                "anchor_cells": ["gamma"],
            },
        ]

        result = collect_candidate_data_from_records(records)

        self.assertEqual(result.qids, ["Q1", "Q2", "Q3", "Q4"])
        self.assertEqual(result.record_count, 2)

    def test_assigns_deduplicated_examples_only_to_all_positive_types(self):
        records = [
            {
                "positive_type_qids": ["Q1", "Q2"],
                "hard_negative_type_qids": ["Q3"],
                "anchor_cells": [" alpha ", "", "alpha", "beta", 7],
            },
            {
                "positive_type_qid": "Q1",
                "positive_type_qids": ["Q1"],
                "hard_negative_type_qids": [],
                "anchor_cells": ["gamma"],
            },
        ]

        result = collect_candidate_data_from_records(records, max_examples=2)

        self.assertEqual(result.examples["Q1"], ["alpha", "beta"])
        self.assertEqual(result.examples["Q2"], ["alpha", "beta"])
        self.assertNotIn("Q3", result.examples)

    def test_rejects_non_positive_example_limit(self):
        with self.assertRaisesRegex(ValueError, "max_examples"):
            collect_candidate_data_from_records([], max_examples=0)

    def test_reads_directory_shards_in_filename_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            later = {
                "positive_type_qid": "Q2",
                "anchor_cells": ["later"],
                "hard_negative_type_qids": [],
            }
            earlier = {
                "positive_type_qid": "Q1",
                "anchor_cells": ["earlier"],
                "hard_negative_type_qids": [],
            }
            (root / "train_shard_0001.jsonl").write_text(
                json.dumps(later) + "\n", encoding="utf-8"
            )
            (root / "train_shard_0000.jsonl").write_text(
                json.dumps(earlier) + "\n", encoding="utf-8"
            )

            records = list(iter_training_records(str(root)))
            result = collect_candidate_data(str(root))

        self.assertEqual(
            [record["positive_type_qid"] for record in records], ["Q1", "Q2"]
        )
        self.assertEqual(result.examples, {"Q1": ["earlier"], "Q2": ["later"]})

    def test_reports_source_and_line_for_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_shard_0000.jsonl"
            path.write_text("{}\nnot-json\n", encoding="utf-8")

            with self.assertRaises(ValueError) as caught:
                list(iter_training_records(str(path)))

        message = str(caught.exception)
        self.assertIn(str(path), message)
        self.assertIn("line 2", message)


if __name__ == "__main__":
    unittest.main()
