import json
import tempfile
import unittest
from pathlib import Path

from WIKIDATA.ontology_parser.build_wikidata_artifact import (
    expected_database_paths,
    parse_args as parse_builder_args,
    write_artifacts,
)
from WIKIDATA.ontology_parser.validate_wikidata_artifact import (
    validate_artifact,
)
from WIKIDATA.ontology_parser.wikidata_artifact import BuildResult


def _record(qid, examples=None):
    return {
        "qid": qid,
        "label": qid.lower(),
        "description": "",
        "aliases": [],
        "parents": [],
        "ancestors": [],
        "children": [],
        "examples": examples or [],
    }


class BuilderCliTests(unittest.TestCase):
    def test_parser_has_hpc_safe_defaults(self):
        args = parse_builder_args(
            [
                "--train-path",
                "/workspace/training",
                "--wd-db",
                "/workspace/wikidata_db",
                "--output",
                "/workspace/training/ontology.jsonl",
            ]
        )

        self.assertEqual(args.max_examples, 10)
        self.assertEqual(args.max_ancestor_depth, 32)

    def test_resolves_explicit_rocksdb_subdirectories(self):
        paths = expected_database_paths(Path("/workspace/wikidata_db"))

        self.assertEqual(
            paths,
            {
                "entity": Path("/workspace/wikidata_db/entities.db"),
                "label": Path("/workspace/wikidata_db/entity_labels.db"),
                "class": Path("/workspace/wikidata_db/classes.db"),
            },
        )

    def test_writes_all_artifacts_with_expected_names(self):
        result = BuildResult(
            records=[_record("Q1", ["alpha"])],
            missing=[{"qid": "Q1", "missing": ["description"]}],
            stats={"record_count": 1},
        )

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "wikidata_type_ontology.jsonl"
            paths = write_artifacts(
                output,
                result,
                {
                    "train_path": "/workspace/training",
                    "wd_db": "/workspace/wikidata_db",
                },
            )

            ontology_lines = paths["ontology"].read_text(
                encoding="utf-8"
            ).splitlines()
            missing_lines = paths["missing"].read_text(
                encoding="utf-8"
            ).splitlines()
            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))

        self.assertEqual(json.loads(ontology_lines[0])["qid"], "Q1")
        self.assertEqual(json.loads(missing_lines[0])["qid"], "Q1")
        self.assertEqual(metadata["record_count"], 1)
        self.assertEqual(metadata["train_path"], "/workspace/training")
        self.assertEqual(
            paths["metadata"].name, "wikidata_type_ontology.meta.json"
        )
        self.assertEqual(
            paths["missing"].name, "wikidata_type_ontology.missing.jsonl"
        )


class ArtifactValidatorTests(unittest.TestCase):
    def _write_training(self, root):
        training = root / "training"
        training.mkdir()
        record = {
            "positive_type_qid": "Q1",
            "positive_type_qids": ["Q1", "Q2"],
            "hard_negative_type_qids": ["Q3"],
            "anchor_cells": ["alpha"],
        }
        (training / "train_shard_0000.jsonl").write_text(
            json.dumps(record) + "\n", encoding="utf-8"
        )
        return training

    def test_accepts_complete_sorted_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            training = self._write_training(root)
            ontology = root / "ontology.jsonl"
            ontology.write_text(
                "\n".join(
                    json.dumps(_record(qid)) for qid in ["Q1", "Q2", "Q3"]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = validate_artifact(str(training), ontology)

        self.assertEqual(summary["training_candidate_count"], 3)
        self.assertEqual(summary["ontology_record_count"], 3)
        self.assertEqual(summary["violations"], 0)

    def test_reports_missing_duplicate_unsorted_and_schema_violations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            training = self._write_training(root)
            ontology = root / "ontology.jsonl"
            invalid = _record("Q2")
            invalid["parents"] = ["Q9"]
            ontology.write_text(
                "\n".join(
                    [
                        json.dumps(invalid),
                        json.dumps(_record("Q1", list(map(str, range(11))))),
                        json.dumps(_record("Q1")),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError) as caught:
                validate_artifact(str(training), ontology)

        message = str(caught.exception)
        self.assertIn("duplicate QID Q1", message)
        self.assertIn("not sorted", message)
        self.assertIn("missing training candidate Q3", message)
        self.assertIn("parents", message)
        self.assertIn("more than 10 examples", message)


if __name__ == "__main__":
    unittest.main()
