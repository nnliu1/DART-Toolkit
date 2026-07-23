import unittest

from WIKIDATA.dart_encoder.mining_utils import (
    build_exclusion_set,
    effective_retrieval_k,
    iter_mining_record_batches,
    select_hard_negatives,
)


class HardNegativeMiningTests(unittest.TestCase):
    def test_cli_propagates_bounded_mining_arguments(self):
        from WIKIDATA.dart_encoder.mine_hard_negative import parse_args

        args = parse_args([
            "--model_path", "model",
            "--train_path", "shards",
            "--ontology_path", "ontology.jsonl",
            "--output_path", "negatives.json",
            "--top_k", "20",
            "--max_rank", "10",
            "--max_length", "192",
        ])
        self.assertEqual(args.top_k, 20)
        self.assertEqual(args.max_rank, 10)
        self.assertEqual(args.max_length, 192)

    def test_effective_retrieval_k_respects_top_k_and_max_rank(self):
        self.assertEqual(effective_retrieval_k(10, 50, 100), 10)
        self.assertEqual(effective_retrieval_k(50, 10, 100), 10)
        with self.assertRaises(ValueError):
            effective_retrieval_k(0, 10, 100)

    def test_record_batches_are_bounded_and_detect_duplicate_keys_before_filtering(self):
        records = [
            {"table_id": "a", "col_index": 0, "positive_type_qid": "Q1"},
            {"table_id": "b", "col_index": 0, "positive_type_qid": "Q1"},
            {"table_id": "c", "col_index": 0, "positive_type_qid": "Q1"},
        ]
        batches = list(iter_mining_record_batches(records, {"Q1"}, 2))
        self.assertEqual([len(batch) for batch in batches], [2, 1])
        duplicates = [
            {"table_id": "a", "col_index": 0, "positive_type_qid": "MISSING"},
            {"table_id": "a", "col_index": 0, "positive_type_qid": "Q1"},
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate sample key"):
            list(iter_mining_record_batches(duplicates, {"Q1"}, 2))

    def test_exclusion_contains_all_positives_and_ignored_types(self):
        record = {
            "positive_type_qid": "Q1",
            "positive_type_qids": ["Q1", "Q2"],
            "ignored_type_qids": ["Q2", "Q3"],
        }
        self.assertEqual(build_exclusion_set(record), {"Q1", "Q2", "Q3"})

    def test_selection_filters_excluded_duplicates_and_uses_safe_fallback(self):
        record = {
            "positive_type_qid": "Q1",
            "positive_type_qids": ["Q1", "Q2"],
            "ignored_type_qids": ["Q3"],
            "hard_negative_type_qids": ["Q2", "Q5", "Q5", "Q6"],
        }
        result = select_hard_negatives(
            ranked_qids=["Q1", "Q2", "Q3", "Q4"],
            ranked_scores=[0.99, 0.9, 0.8, 0.7],
            record=record,
            ontology_qids={"Q1", "Q2", "Q3", "Q4", "Q5", "Q6"},
            n_hard_negs=3,
            min_rank=1,
            max_rank=4,
            min_score=0.1,
        )
        self.assertEqual(result, ["Q4", "Q5", "Q6"])


if __name__ == "__main__":
    unittest.main()
