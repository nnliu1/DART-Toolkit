"""Export runtime ontology classes in the format expected by DART."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Iterable
from urllib.parse import unquote, urlsplit

from rdflib import Graph, Literal, OWL, RDF, RDFS, URIRef


def build_dart_ontology(
    source_path: str | Path, output_path: str | Path
) -> int:
    """Exports named ontology classes as deterministic DART passages.

    Args:
        source_path: OWL, RDF, or Turtle ontology file.
        output_path: Destination JSON file.

    Returns:
        Number of exported classes.

    Raises:
        ValueError: If the ontology contains no named classes.
    """
    graph = Graph()
    graph.parse(Path(source_path))
    class_iris = sorted(str(value) for value in _named_classes(graph))
    if not class_iris:
        raise ValueError("Ontology contains no named classes")

    concepts = {}
    for iri in class_iris:
        resource = URIRef(iri)
        concepts[iri] = {
            "label": _label(graph, resource),
            "description": _preferred_literal(graph.objects(resource, RDFS.comment)),
            "parents": _parent_labels(graph, resource),
        }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"concepts": concepts}, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return len(concepts)


def _named_classes(graph: Graph) -> set[URIRef]:
    classes = {
        subject
        for class_type in (OWL.Class, RDFS.Class)
        for subject in graph.subjects(RDF.type, class_type)
        if isinstance(subject, URIRef)
    }
    classes.update(
        value
        for subject, parent in graph.subject_objects(RDFS.subClassOf)
        for value in (subject, parent)
        if isinstance(value, URIRef)
    )
    return classes


def _label(graph: Graph, resource: URIRef) -> str:
    return _preferred_literal(graph.objects(resource, RDFS.label)) or _humanize_iri(
        str(resource)
    )


def _preferred_literal(values: Iterable[object]) -> str:
    literals = sorted(
        (value for value in values if isinstance(value, Literal)),
        key=lambda value: (
            0 if value.language == "en" else 1 if value.language is None else 2,
            str(value),
        ),
    )
    return str(literals[0]) if literals else ""


def _parent_labels(graph: Graph, resource: URIRef) -> list[str]:
    parents = {
        _label(graph, parent)
        for parent in graph.objects(resource, RDFS.subClassOf)
        if isinstance(parent, URIRef)
    }
    return sorted(parents)


def _humanize_iri(iri: str) -> str:
    parsed = urlsplit(iri)
    local_name = unquote(parsed.fragment or parsed.path.rsplit("/", 1)[-1])
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", local_name)
    return spaced.replace("_", " ").replace("-", " ").strip()
