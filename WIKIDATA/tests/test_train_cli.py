import tempfile
import unittest
from pathlib import Path

from WIKIDATA.dart_encoder.train import (
    is_better_checkpoint,
    load_validation_samples,
    parse_args,
)


class TrainCliTests(unittest.TestCase):
    def test_accepts_directory_or_glob_path_as_plain_input_spec(self):
        args = parse_args([
            "--train_path", "data/train_shard_*.jsonl",
            "--val_path", "data/val",
            "--ontology_path", "ontology.jsonl",
            "--output_dir", "output",
        ])
        self.assertEqual(args.train_path, "data/train_shard_*.jsonl")
        self.assertEqual(args.val_path, "data/val")
        self.assertEqual(args.min_majority_ratio, 0.7)
        self.assertEqual(args.min_type_count, 3)

    def test_requires_validation_path(self):
        with self.assertRaises(SystemExit):
            parse_args([
                "--train_path", "data/train",
                "--ontology_path", "ontology.jsonl",
                "--output_dir", "output",
            ])

    def test_first_checkpoint_is_always_best(self):
        self.assertTrue(is_better_checkpoint(0.0, None))

    def test_equal_recall_keeps_earlier_checkpoint(self):
        self.assertFalse(is_better_checkpoint(0.5, 0.5))

    def test_rejects_empty_validation_dataset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "val.jsonl"
            path.touch()

            with self.assertRaisesRegex(ValueError, "Validation dataset is empty"):
                load_validation_samples(str(path))


if __name__ == "__main__":
    unittest.main()
