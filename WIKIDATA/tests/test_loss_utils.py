import unittest

from WIKIDATA.dart_encoder.loss_utils import build_positive_mask_rows


class PositiveMaskTests(unittest.TestCase):
    def test_marks_owned_positives_and_same_qid_duplicates_across_batch(self):
        mask = build_positive_mask_rows(
            pos_counts=[2, 1],
            positive_qid_sets=[["Q1", "Q2"], ["Q1"]],
            positive_qids_flat=["Q1", "Q2", "Q1"],
        )
        self.assertEqual(mask, [
            [True, True, True],
            [True, False, True],
        ])

    def test_rejects_inconsistent_flattened_positive_count(self):
        with self.assertRaises(ValueError):
            build_positive_mask_rows([2], [["Q1", "Q2"]], ["Q1"])


if __name__ == "__main__":
    unittest.main()
