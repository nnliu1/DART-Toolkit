import json
import tempfile
import unittest
from pathlib import Path

from WIKIDATA.ontology_parser.wikidata_artifact import (
    CandidateData,
    build_artifact,
    collect_candidate_data,
    collect_candidate_data_from_records,
    iter_training_records,
)


class FakeClass:
    def __init__(self, parents):
        self.parents = parents


class FakeLabel:
    def __init__(self, label):
        self.label = label


class FakeEntity:
    def __init__(self, label=None, description=None, aliases=None):
        self.label = label
        self.description = description
        self.aliases = aliases


class FakeMultilingual:
    def __init__(self, lang2value):
        self.lang2value = lang2value


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


class MetadataNormalizationTests(unittest.TestCase):
    def test_normalizes_mapping_metadata_and_removes_primary_label_alias(self):
        entity_db = {
            "Q1": {
                "label": {"en": "child"},
                "description": {"en": "a child type", "de": "Kindtyp"},
                "aliases": {"en": ["offspring", "Child", "offspring"]},
            }
        }

        result = build_artifact(
            CandidateData(["Q1"], {"Q1": ["Alice"]}, 1),
            entity_db,
            {"Q1": "child"},
            {"Q1": FakeClass([])},
        )

        record = result.records[0]
        self.assertEqual(record["label"], "child")
        self.assertEqual(record["description"], "a child type")
        self.assertEqual(record["aliases"], ["offspring"])
        self.assertEqual(record["examples"], ["Alice"])
        self.assertEqual(result.missing, [])

    def test_accepts_label_and_entity_objects(self):
        entity_db = {
            "Q1": FakeEntity(
                label="entity fallback",
                description="object description",
                aliases=["first alias", "second alias"],
            )
        }

        result = build_artifact(
            CandidateData(["Q1"], {}, 1),
            entity_db,
            {"Q1": FakeLabel("label database value")},
            {"Q1": FakeClass([])},
        )

        self.assertEqual(result.records[0]["label"], "label database value")
        self.assertEqual(result.records[0]["description"], "object description")
        self.assertEqual(
            result.records[0]["aliases"], ["first alias", "second alias"]
        )

    def test_accepts_lang2value_wrappers(self):
        entity_db = {
            "Q1": FakeEntity(
                label=FakeMultilingual({"en": "wrapped label"}),
                description=FakeMultilingual({"en": "wrapped description"}),
                aliases=FakeMultilingual({"en": ["wrapped alias"]}),
            )
        }

        result = build_artifact(
            CandidateData(["Q1"], {}, 1),
            entity_db,
            {},
            {"Q1": FakeClass([])},
        )

        self.assertEqual(result.records[0]["label"], "wrapped label")
        self.assertEqual(result.records[0]["description"], "wrapped description")
        self.assertEqual(result.records[0]["aliases"], ["wrapped alias"])

    def test_uses_qid_fallback_and_reports_missing_fields(self):
        result = build_artifact(
            CandidateData(["Q9"], {}, 1),
            {},
            {},
            {},
        )

        self.assertEqual(
            result.records[0],
            {
                "qid": "Q9",
                "label": "Q9",
                "description": "",
                "aliases": [],
                "parents": [],
                "ancestors": [],
                "children": [],
                "examples": [],
            },
        )
        self.assertEqual(
            result.missing,
            [
                {
                    "qid": "Q9",
                    "missing": [
                        "entity",
                        "label",
                        "description",
                        "aliases",
                        "class",
                    ],
                }
            ],
        )


class HierarchyConstructionTests(unittest.TestCase):
    def test_builds_cycle_safe_breadth_first_ancestors_and_children(self):
        classes = {
            "Q1": FakeClass(["Q3", "Q2"]),
            "Q2": FakeClass(["Q4"]),
            "Q3": FakeClass(["Q4"]),
            "Q4": FakeClass(["Q1"]),
        }
        labels = {
            "Q1": "one",
            "Q2": "two",
            "Q3": "three",
            "Q4": "four",
        }

        result = build_artifact(
            CandidateData(["Q1", "Q2"], {}, 1),
            {},
            labels,
            classes,
        )

        q1, q2 = result.records
        self.assertEqual(
            q1["parents"],
            [
                {"qid": "Q2", "label": "two"},
                {"qid": "Q3", "label": "three"},
            ],
        )
        self.assertEqual(q1["ancestors"], ["Q2", "Q3", "Q4"])
        self.assertEqual(q1["children"], [])
        self.assertEqual(q2["children"], ["Q1"])

    def test_limits_ancestor_traversal_by_depth(self):
        classes = {
            "Q1": FakeClass(["Q2"]),
            "Q2": FakeClass(["Q3"]),
            "Q3": FakeClass(["Q4"]),
            "Q4": FakeClass([]),
        }

        result = build_artifact(
            CandidateData(["Q1"], {}, 1),
            {},
            {},
            classes,
            max_ancestor_depth=2,
        )

        self.assertEqual(result.records[0]["ancestors"], ["Q2", "Q3"])

    def test_children_exclude_types_outside_candidate_universe(self):
        classes = {
            "Q1": FakeClass(["Q2"]),
            "Q3": FakeClass(["Q2"]),
        }

        result = build_artifact(
            CandidateData(["Q1", "Q2"], {}, 1),
            {},
            {},
            classes,
        )

        q2 = next(record for record in result.records if record["qid"] == "Q2")
        self.assertEqual(q2["children"], ["Q1"])

    def test_rejects_non_positive_ancestor_depth(self):
        with self.assertRaisesRegex(ValueError, "max_ancestor_depth"):
            build_artifact(
                CandidateData(["Q1"], {}, 1),
                {},
                {},
                {},
                max_ancestor_depth=0,
            )


if __name__ == "__main__":
    unittest.main()
