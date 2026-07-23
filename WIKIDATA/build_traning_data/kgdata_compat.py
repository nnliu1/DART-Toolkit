"""Compatibility helpers for kgdata database factory API changes."""

from __future__ import annotations

from pathlib import Path


def resolve_class_db_factory(db_module):
    """Return the class DB factory across supported kgdata releases."""
    for name in ("get_wdclass_db", "get_class_db"):
        factory = getattr(db_module, name, None)
        if factory is not None:
            return factory
    raise ImportError(
        "kgdata.wikidata.db exposes neither get_wdclass_db nor get_class_db"
    )


def open_wikidata_databases(
    root,
    *,
    entity_factory,
    label_factory,
    class_factory,
):
    """Open kgdata Wikidata RocksDB subdatabases without taking write locks."""
    root = Path(root)
    options = {"create_if_missing": False, "read_only": True}
    entity_db = entity_factory(root / "entities.db", **options)
    label_db = label_factory(root / "entity_labels.db", **options)
    class_db = class_factory(root / "classes.db", **options)
    return entity_db, label_db, class_db


def resolve_label_entry(entry):
    """Normalize label DB values across kgdata releases."""
    if isinstance(entry, str):
        label = entry
    elif isinstance(entry, dict):
        label = entry.get("label", "")
    else:
        label = getattr(entry, "label", "")
    if isinstance(label, str) and len(label) >= 2:
        if label.startswith('"') and label.endswith('"'):
            label = label[1:-1]
    return label or None
