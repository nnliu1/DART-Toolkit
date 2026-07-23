import unittest

from WIKIDATA.dart_encoder.train import parse_args


class TrainCliTests(unittest.TestCase):
    def test_accepts_directory_or_glob_path_as_plain_input_spec(self):
        args = parse_args([
            "--train_path", "data/train_shard_*.jsonl",
            "--ontology_path", "ontology.jsonl",
            "--output_dir", "output",
        ])
        self.assertEqual(args.train_path, "data/train_shard_*.jsonl")
        self.assertEqual(args.min_majority_ratio, 0.7)
        self.assertEqual(args.min_type_count, 3)


if __name__ == "__main__":
    unittest.main()
