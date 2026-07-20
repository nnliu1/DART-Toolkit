import random
from dataclasses import dataclass

from WIKIDATA.build_traning_data.cta_labeling import (
    HardNegativeMiner,
    collect_ancestors,
    infer_column_types,
)


class FakeValue:
    def __init__(self, qid):
        self.qid = qid

    def as_entity_id_safe(self):
        return self.qid


@dataclass
class FakeStatement:
    value: FakeValue


@dataclass
class FakeEntity:
    props: dict


@dataclass
class FakeClass:
    parents: list


def entity(*types):
    return FakeEntity({"P31": [FakeStatement(FakeValue(t)) for t in types]})


def test_entity_coverage_deduplicates_entities_and_per_entity_types():
    entity_db = {
        "Q1": entity("T1", "T1", "T2"),
        "Q2": entity("T1"),
        "Q3": entity("T1"),
    }

    result = infer_column_types(
        ["Q1", "Q1", "Q2", "Q3"],
        entity_db,
        majority_threshold=2 / 3,
        positive_coverage_threshold=1 / 3,
    )

    assert result is not None
    assert result.primary_qid == "T1"
    assert result.primary_coverage == 1.0
    assert result.num_typed_entities == 3
    assert result.entity_counts == {"T1": 3, "T2": 1}
    assert result.positive_qids == ["T1", "T2"]
    assert result.support == {"T1": 1.0, "T2": 1 / 3}


def test_primary_type_tie_breaks_by_qid():
    result = infer_column_types(
        ["Q1", "Q2"],
        {"Q1": entity("T2"), "Q2": entity("T1")},
        majority_threshold=0.5,
        positive_coverage_threshold=0.5,
    )

    assert result is not None
    assert result.primary_qid == "T1"
    assert result.positive_qids == ["T1", "T2"]


def test_rejects_when_primary_entity_coverage_is_below_threshold():
    result = infer_column_types(
        ["Q1", "Q2", "Q3"],
        {"Q1": entity("T1"), "Q2": entity("T2"), "Q3": entity("T3")},
        majority_threshold=0.5,
        positive_coverage_threshold=0.25,
    )

    assert result is None


def test_collect_ancestors_stops_at_depth_and_handles_missing_classes():
    class_db = {
        "T1": FakeClass(["P1"]),
        "P1": FakeClass(["P2"]),
        "P2": FakeClass(["P3"]),
    }

    assert collect_ancestors(["T1", "UNKNOWN"], class_db, max_depth=2) == {
        "P1",
        "P2",
    }


def test_hard_negative_miner_never_returns_excluded_types():
    class_db = {
        "T1": FakeClass(["P"]),
        "T2": FakeClass(["P"]),
        "T3": FakeClass(["P"]),
        "T4": FakeClass([]),
    }
    miner = HardNegativeMiner(
        class_db=class_db,
        global_type_pool=["T1", "T2", "T3", "T4"],
        num_negatives=3,
        rng=random.Random(42),
    )

    negatives = miner.mine("T1", excluded_qids={"T2", "T4"})

    assert negatives == ["T3"]
    assert not ({"T1", "T2", "T4"} & set(negatives))
