import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from data.split_training_data import (
    is_validation_group,
    page_key,
    parse_args,
    resolve_input_shards,
    split_dataset,
)


class PageAssignmentTests(unittest.TestCase):
    def test_resolves_only_sorted_training_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in (
                "train_shard_0001.jsonl",
                "wikidata_type_ontology.jsonl",
                "train_shard_0000.jsonl",
            ):
                (root / name).write_text("{}\n", encoding="utf-8")

            paths = resolve_input_shards(root)

        self.assertEqual(
            [path.name for path in paths],
            ["train_shard_0000.jsonl", "train_shard_0001.jsonl"],
        )

    def test_page_key_removes_only_table_number(self):
        key, malformed = page_key(
            "https://en.wikipedia.org/wiki/Example?table_no=4"
        )

        self.assertEqual(key, "https://en.wikipedia.org/wiki/Example")
        self.assertFalse(malformed)

    def test_page_key_reports_malformed_identifier(self):
        key, malformed = page_key("")

        self.assertEqual(key, "")
        self.assertTrue(malformed)

    def test_assignment_is_deterministic_and_page_based(self):
        key = "https://en.wikipedia.org/wiki/Example"

        decisions = [
            is_validation_group(key, val_ratio=0.1, seed=42)
            for _ in range(5)
        ]

        self.assertEqual(decisions, [decisions[0]] * 5)
        self.assertEqual(
            decisions[0],
            is_validation_group(key, val_ratio=0.1, seed=42),
        )

    def test_assignment_rejects_invalid_ratio(self):
        for ratio in (0.0, 1.0, -0.1, 1.1):
            with self.subTest(ratio=ratio):
                with self.assertRaisesRegex(ValueError, "val_ratio"):
                    is_validation_group("page", val_ratio=ratio, seed=42)


class DatasetSplitTests(unittest.TestCase):
    @staticmethod
    def _page_for_partition(validation):
        for index in range(10_000):
            page = "https://en.wikipedia.org/wiki/Page_{0}".format(index)
            if is_validation_group(page, 0.1, 42) is validation:
                return page
        raise AssertionError("Could not find a page for requested partition")

    @staticmethod
    def _record(table_id, primary, positives=None):
        return {
            "anchor_header": "Name",
            "anchor_cells": ["example"],
            "positive_type_qid": primary,
            "positive_type_qids": positives or [primary],
            "hard_negative_type_qids": ["Q9"],
            "table_id": table_id,
        }

    @staticmethod
    def _read_records(directory):
        output = []
        for path in sorted(directory.glob("train_shard_*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                output.extend(json.loads(line) for line in handle if line.strip())
        return output

    def test_splits_records_exactly_once_without_page_or_table_leakage(self):
        train_page = self._page_for_partition(False)
        val_page = self._page_for_partition(True)
        records = [
            self._record(train_page + "?table_no=1", "Q1"),
            self._record(train_page + "?table_no=2", "Q1", ["Q1", "Q2"]),
            self._record(val_page + "?table_no=1", "Q1"),
            self._record(val_page + "?table_no=2", "Q3"),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "train_shard_0000.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            output = root / "split"

            metadata = split_dataset(
                source,
                output,
                val_ratio=0.1,
                seed=42,
                shard_size=2,
            )
            train_records = self._read_records(output / "train")
            val_records = self._read_records(output / "val")
            persisted = json.loads(
                (output / "split_metadata.json").read_text(encoding="utf-8")
            )

        self.assertCountEqual(train_records + val_records, records)
        self.assertEqual(len(train_records), 2)
        self.assertEqual(len(val_records), 2)
        self.assertEqual(metadata, persisted)
        self.assertEqual(metadata["page_intersection_count"], 0)
        self.assertEqual(metadata["table_intersection_count"], 0)
        self.assertEqual(metadata["validation_seen_positive_type_count"], 1)
        self.assertEqual(metadata["validation_unseen_positive_type_count"], 1)
        self.assertEqual(metadata["validation_unseen_positive_types"], ["Q3"])
        self.assertEqual(metadata["train_shard_count"], 1)
        self.assertEqual(metadata["validation_shard_count"], 1)

    def test_respects_output_shard_size(self):
        train_page = self._page_for_partition(False)
        val_page = self._page_for_partition(True)
        records = [
            self._record(
                "{0}?table_no={1}".format(train_page, index),
                "Q1",
            )
            for index in range(5)
        ]
        records.append(self._record(val_page + "?table_no=1", "Q2"))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "train_shard_0000.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            output = root / "split"

            split_dataset(source, output, shard_size=2)
            shard_sizes = [
                len(path.read_text(encoding="utf-8").splitlines())
                for path in sorted((output / "train").glob("*.jsonl"))
            ]

        self.assertEqual(shard_sizes, [2, 2, 1])

    def test_refuses_to_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "train_shard_0000.jsonl").write_text(
                json.dumps(
                    self._record(
                        self._page_for_partition(False) + "?table_no=1",
                        "Q1",
                    )
                )
                + "\n"
                + json.dumps(
                    self._record(
                        self._page_for_partition(True) + "?table_no=1",
                        "Q2",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "split"
            output.mkdir()

            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                split_dataset(source, output)


class SplitCliTests(unittest.TestCase):
    def test_parser_defaults_and_overwrite_flag(self):
        args = parse_args(
            [
                "--input-dir",
                "source",
                "--output-dir",
                "split",
                "--overwrite",
            ]
        )

        self.assertEqual(args.val_ratio, 0.1)
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.shard_size, 50_000)
        self.assertTrue(args.overwrite)

    def test_standalone_help_succeeds(self):
        repository = Path(__file__).resolve().parents[2]
        script = repository / "data" / "split_training_data.py"

        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=str(repository),
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--input-dir", result.stdout)
        self.assertIn("--output-dir", result.stdout)
        self.assertIn("--overwrite", result.stdout)


if __name__ == "__main__":
    unittest.main()
