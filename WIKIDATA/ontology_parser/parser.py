from __future__ import annotations
 
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import sys
from collections import Counter, defaultdict

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-7s  %(message)s",
    stream = sys.stdout,
)
log = logging.getLogger(__name__)

"""
construct description test for ontology concepts:

 concept_id
 label
 aliases: rdfs:label extras + skos:altLabel
 description : rdfs:commemt
 parent: rdfs:subClassOf
 domain: rdfs:domain
 range: rdfs:range

"""

ONTO_PATH = "/home/iai/dg3485/CTA/iswc26/data/ontology/dbpedia.owl",
OUT_DIR = "iswc/2026/preprocess_ontology"
ONTO_NAME = "dbo"
# TRAIN_PATH = "/home/iai/dg3485/CTA/iswc26/data/train_bridge/no_col/training_pairs.json"
# TRAIN_HEADER_TOPK = 5
USE_WIKIPEDIA_API: bool = True
WIKIPEDIA_API_SLEEP: float = 0.5
WIKIPEDIA_MAX_CHARS: int = 300
USE_WIKIPEDIA_REDIRECTS: bool = True
WIKIPEDIA_REDIRECTS_TOPK: int = 3


_FMT_MAP: Dict[str, str] = {
    ".ttl":  "turtle",
    ".owl":  "xml",
    ".rdf":  "xml",
    ".xml":  "xml",
    ".n3":   "n3",
    ".nt":   "nt",
    ".trig": "trig",
    ".nq":   "nquads",
    ".jsonld": "json-ld",
}

class Concept:
    __slots__ = (
        "uri", "label", "comment", "synonyms",
        "parents", "children",
        "concept_id", "aliases", "description",
        "parent_label", "domain_label", "range_label",
        "children_labels",   # direct child class labels (for retrieval context)
        "ancestor_chain",    # [parent_label, grandparent_label, ...] up to root
    )
 
    def __init__(
        self,
        uri:      str,
        label:    str,
        comment:  Optional[str]  = None,
        synonyms: Optional[List[str]] = None,
    ) -> None:
        self.uri      = uri
        self.label    = label
        self.comment  = comment
        self.synonyms: List[str] = synonyms or []
        self.parents:  List[str] = []
        self.children: List[str] = []
 
        # TypeIndex fields
        self.concept_id:   str           = uri
        self.aliases:      List[str]     = synonyms or []
        self.description:  Optional[str] = comment
        self.parent_label:   Optional[str]  = None
        self.domain_label:   Optional[str]  = None
        self.range_label:    Optional[str]  = None
        self.children_labels: List[str]     = []
        self.ancestor_chain:  List[str]     = []  # filled after full graph built
 
 
class OntologyCorpus:
    def __init__(
        self,
        concepts: Dict[str, Concept],
        triplets: List[Tuple[str, str, str]],
    ) -> None:
        self.concepts = concepts
        self.triplets = triplets
        self.paths: List[List[str]] = self._build_paths()
 
    def _build_paths(self) -> List[List[str]]:
        """
        Ancestor chain per concept:
            [concept_uri, parent_uri, grandparent_uri, ..., root_uri]
        Cycles are broken via visited set.
        """
        paths: List[List[str]] = []
        for uri in self.concepts:
            chain:   List[str] = []
            visited: set[str]  = set()
            current = uri
            while current and current not in visited:
                chain.append(current)
                visited.add(current)
                parents = self.concepts[current].parents
                current = parents[0] if parents else None
            paths.append(chain)
        return paths
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Internal rdflib helpers
# ─────────────────────────────────────────────────────────────────────────────
 
def _uri_local_name(uri: str) -> str:
    for sep in ("#", "/"):
        idx = uri.rfind(sep)
        if idx != -1 and idx < len(uri) - 1:
            return uri[idx + 1:]
    return uri
 
 
def _pick_label(g, ref, RDFS_LBL, Literal, uri: str) -> str:
    """
    Return the best English rdfs:label for a class URI.
 
    Preference order:
      1. Literal with lang tag "en", "en-us", or "en-gb"
      2. Literal with NO lang tag (assumed English — common in DBpedia OWL)
      3. URI local name as last resort
 
    Non-English literals (de, fr, ar, zh, ...) are always skipped,
    even as fallback.  This prevents non-English labels from polluting
    the concept text that is fed to SBERT.
    """
    en_tagged  = None   # lang = en / en-us / en-gb
    untagged   = None   # lang = None  (accept as English-compatible)
 
    for obj in g.objects(ref, RDFS_LBL):
        if not isinstance(obj, Literal):
            continue
        lang = getattr(obj, "language", None)
        if lang is not None and not lang.lower().startswith("en"):
            continue    # skip any non-English tagged literal
        text = str(obj).strip()
        if not text:
            continue
        if lang is not None:          # en / en-us / en-gb
            if en_tagged is None:
                en_tagged = text
        else:                         # no lang tag
            if untagged is None:
                untagged = text
 
    return en_tagged or untagged or _uri_local_name(uri)
 
 
def _is_clean_english(text: str) -> bool:
    """
    Return True if text looks like clean English prose.
    Rejects:
      - strings containing http:// or https:// URIs
      - strings where more than 15% of characters are non-ASCII
        (catches Arabic, Urdu, Chinese, etc.)
      - very short strings (< 10 chars) that are unlikely to be useful
    """
    if not text or len(text) < 10:
        return False
    if "http://" in text or "https://" in text:
        return False
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    if non_ascii / len(text) > 0.15:
        return False
    return True
 
 
def _pick_comment(g, ref, RDFS_CMT, Literal) -> Optional[str]:
    """Return a clean English rdfs:comment only.
    Skips non-English literals, URIs in text, and non-ASCII-heavy strings."""
    for obj in g.objects(ref, RDFS_CMT):
        if not isinstance(obj, Literal):
            continue
        lang = getattr(obj, "language", None)
        if lang is not None and not lang.lower().startswith("en"):
            continue
        text = str(obj).strip()
        if _is_clean_english(text):
            return text
    return None
 
 
def _pick_synonyms(g, ref, RDFS_LBL, Literal, primary_label: str) -> List[str]:
    """Return English-only alternate labels (rdfs:label + skos:altLabel).
    Literals with no language tag are accepted as English-compatible.
    The primary label is excluded from the result."""
    try:
        from rdflib.namespace import SKOS
        skos_alt = SKOS.altLabel
    except ImportError:
        from rdflib import URIRef as _URIRef
        skos_alt = _URIRef("http://www.w3.org/2004/02/skos/core#altLabel")
 
    synonyms: List[str] = []
    seen = {primary_label}
    for pred in (RDFS_LBL, skos_alt):
        for obj in g.objects(ref, pred):
            if not isinstance(obj, Literal):
                continue
            lang = getattr(obj, "language", None)
            if lang is not None and not lang.lower().startswith("en"):
                continue        # skip non-English
            val = str(obj)
            if val not in seen:
                synonyms.append(val)
                seen.add(val)
    return synonyms
 
# def _camel_split(label: str) -> List[str]:
#     """
#     Generate additional English alias variants from a CamelCase label.

#     """
#     import re
#     # split on transitions: lowercase→Uppercase or sequence-of-Uppers→Uppercase+lower
#     words = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|)", label)
#     if len(words) < 2:
#         return []
#     lower_words = [w.lower() for w in words]
#     variants = [
#         " ".join(lower_words),           # "baseball team"
#         "_".join(lower_words),           # "baseball_team"
#     ]
#     # also add individual meaningful words (skip single-char and stop words)
#     _STOP = {"a", "an", "the", "of", "in", "by", "for", "and", "or"}
#     for w in lower_words:
#         if len(w) > 2 and w not in _STOP:
#             variants.append(w)
#     return list(dict.fromkeys(variants))  # dedup, preserve order
 
 
# def mine_headers_from_train(
#     train_paths: List[str],
#     topk: int = 5,
# ) -> Dict[str, List[str]]:
#     """
#     Scan training JSON files and collect the most frequent column headers
#     per ontology URI.

#     """
#     if isinstance(train_paths, str):
#         train_paths = [train_paths]

#     uri_to_headers: Dict[str, Counter] = defaultdict(Counter)
 
#     for fpath in train_paths:
#         p = Path(fpath)
#         if not p.exists():
#             log.warning("TRAIN_PATH not found, skipping: %s", fpath)
#             continue
#         with open(fpath, encoding="utf-8") as f:
#             samples = json.load(f)
#         for s in samples:
#             uri    = (s.get("meta") or {}).get("gt_uri", "")
#             header = (s.get("table") or {}).get("header", "")
#             if uri and header and header.strip().lower() not in ("", "unknown"):
#                 uri_to_headers[uri][header.strip().lower()] += 1
 
#     result: Dict[str, List[str]] = {}
#     for uri, counter in uri_to_headers.items():
#         result[uri] = [h for h, _ in counter.most_common(topk)]
 
#     log.info(
#         "Header mining done: %d URIs with training headers (topk=%d)",
#         len(result), topk,
#     )
#     return result

## wikidata descriptions for enrich concept context
 
def extract_wikidata_qids_from_owl(owl_path: str) -> Dict[str, str]:
    """
    Parse the DBpedia OWL file and extract owl:equivalentClass mappings
    to Wikidata QIDs.
 
    Returns
    -------
    dict  dbo_uri → wikidata_qid  (e.g. "http://dbpedia.org/ontology/City" → "Q515")
    """
    try:
        import rdflib
        from rdflib import OWL, URIRef
    except ImportError:
        raise ImportError("rdflib is required: pip install rdflib")
 
    WIKIDATA_PREFIX = "http://www.wikidata.org/entity/"
 
    g = rdflib.Graph()
    fmt = _FMT_MAP.get(Path(owl_path).suffix.lower(), "xml")
    g.parse(owl_path, format=fmt)
 
    qid_map: Dict[str, str] = {}
    for s, o in g.subject_objects(OWL.equivalentClass):
        if not (isinstance(s, URIRef) and isinstance(o, URIRef)):
            continue
        dbo_uri = str(s)
        eq_uri  = str(o)
        if eq_uri.startswith(WIKIDATA_PREFIX):
            qid = eq_uri[len(WIKIDATA_PREFIX):]   # e.g. "Q515"
            qid_map[dbo_uri] = qid
 
    log.info(
        "extract_wikidata_qids_from_owl: %d dbo→QID mappings found in %s",
        len(qid_map), owl_path,
    )
    return qid_map
 
 
def fetch_wikipedia_descriptions(
    uris:      List[str],
    sleep:     float = 0.2,
    max_chars: int   = 300,
    fetch_redirects: bool = True,
    redirects_topk: int = 3,
) -> Dict[str, str]:
    """
    Fetch English descriptions from Wikipedia REST API for a list of
    ontology class URIs.
    """
    import time
    import urllib.request
    import urllib.parse
    import json as _json
 
    SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
    REDIRECT_URL = "https://en.wikipedia.org/w/api.php"
    HEADERS  = {
        "User-Agent": "build_concept_text/1.0 (ontology enrichment)",
        "Accept":     "application/json",
    }
 
    _REDIRECT_SKIP = {
        "wikipedia", "redirect", "disambiguation",
        "talk", "user", "template", "category",
    }


    def _first_sentence(text: str, maxlen: int) -> str:
        """Extract first sentence, truncate cleanly at word boundary."""
        # try to find sentence end
        for sep in (". ", ".\n"):
            idx = text.find(sep)
            if 0 < idx < maxlen:
                return text[:idx + 1].strip()
        # no sentence boundary found — truncate at last space before maxlen
        truncated = text[:maxlen]
        last_space = truncated.rfind(" ")
        if last_space > maxlen // 2:
            return truncated[:last_space].strip() + "..."
        return truncated.strip()
 
    def _fetch_summary(title: str):
        """
        Rejects disambiguation pages and pages without a clean extract.
        """
        encoded = urllib.parse.quote(title, safe="")
        url     = SUMMARY_URL + encoded
        req     = urllib.request.Request(
            url, headers={**HEADERS, "Accept-Language": "en"}
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = _json.loads(resp.read())
 
            # reject non-English responses
            if data.get("lang", "en") != "en":
                return None, None
 
            # reject disambiguation and stub pages — they have no clean definition
            page_type = data.get("type", "standard")
            if page_type in ("disambiguation", "no-extract"):
                log.debug("Skipping %s page: %r", page_type, title)
                return None, None
 
            extract = data.get("extract", "")
            desc    = _first_sentence(extract, max_chars) if (
                extract and _is_clean_english(extract)
            ) else None
            canonical = data.get("titles", {}).get("canonical") or data.get("title")
            return desc, canonical
        except Exception as exc:
            log.debug("Wikipedia summary failed for %r: %s", title, exc)
            return None, None


    def _fetch_redirects(canonical_title: str) -> List[str]:
        """
        Query Wikipedia API for all pages that redirect to canonical_title.
        Returns a list of clean English redirect titles (max redirects_topk).
        """
        params = urllib.parse.urlencode({
            "action":   "query",
            "titles":   canonical_title,
            "prop":     "redirects",
            "rdlimit":  50,          # fetch up to 50, filter down to topk
            "rdnamespace": 0,        # main namespace only
            "format":   "json",
        })
        url = REDIRECT_URL + "?" + params
        req = urllib.request.Request(url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = _json.loads(resp.read())
        except Exception as exc:
            log.debug("Wikipedia redirects failed for %r: %s", canonical_title, exc)
            return []
 
        aliases: List[str] = []
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            for rd in page.get("redirects", []):
                title = rd.get("title", "")
                # skip if empty or too short
                if not title or len(title) < 3:
                    continue
                # skip disambiguation redirects like "President (title)",
                # "Novel (book)" — parenthesised titles are disambiguation entries
                if "(" in title and ")" in title:
                    continue
                # skip redirects starting with "List of" or "The " — not synonyms
                title_lower = title.lower()
                if title_lower.startswith(("list of", "the ", "a ", "an ")):
                    continue
                # skip system/meta page indicators
                if any(skip in title_lower for skip in _REDIRECT_SKIP):
                    continue
                # skip pure numbers or years
                if title.isdigit():
                    continue
                # skip non-ASCII-heavy titles (foreign language redirects)
                if not _is_clean_english(title):
                    continue
                # skip titles that are just the plural of the canonical title
                # e.g. "Enzymes" when canonical is "Enzyme" — low value
                # (keep plurals only if they differ substantially)
                clean = title.strip()
                if not clean or clean.lower() == canonical_title.lower():
                    continue
                # skip if only difference is trailing "s" (simple plural)
                if clean.lower() == canonical_title.lower() + "s":
                    continue
                aliases.append(clean)
                if len(aliases) >= redirects_topk:
                    break
        return aliases
 
    results: Dict[str, Dict] = {}
    n_total   = len(uris)
    n_desc    = 0
    n_aliases = 0
 
    log.info(
        "Fetching Wikipedia data for %d URIs (redirects=%s) ...",
        n_total, fetch_redirects,
    )
 
    for i, uri in enumerate(uris):
        if i > 0 and i % 50 == 0:
            log.info(
                "  %d / %d  desc=%d  aliases=%d",
                i, n_total, n_desc, n_aliases,
            )
 
        local = _uri_local_name(uri)
        desc, canonical = _fetch_summary(local)
 
        # fallback: CamelCase split
        if desc is None and len(local) > 4:
            import re
            words = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\b)", local)
            if len(words) > 1:
                title2 = " ".join(words)
                desc, canonical = _fetch_summary(title2)
 
        aliases: List[str] = []
        if fetch_redirects and canonical:
            time.sleep(sleep)   # extra sleep before redirect call
            aliases = _fetch_redirects(canonical)
 
        results[uri] = {"description": desc, "aliases": aliases}
 
        if desc:
            n_desc += 1
        if aliases:
            n_aliases += 1
 
        time.sleep(sleep)
 
    log.info(
        "fetch_wikipedia_descriptions done: "
        "desc=%d/%d  aliases=%d/%d URIs",
        n_desc, n_total, n_aliases, n_total,
    )
    return results

## main functions 
def load_ontology(
        path: str,
        abstracts: Optional[Dict[str, str]] = None,
        ) -> OntologyCorpus:
    # import rdflib
    try:
        import rdflib
        from rdflib import RDF, RDFS, OWL, Literal, URIRef
    except ImportError:
        raise ImportError("rdflib is required: pip install rdflib")
 
    # detect format and parse graph
    ext = Path(path).suffix.lower()
    fmt = _FMT_MAP.get(ext)
    g   = rdflib.Graph()
 
    if fmt is None:
        log.warning(
            "Unknown extension %r — attempting turtle then xml parsing.", ext
        )
        for attempt_fmt in ("turtle", "xml"):
            try:
                g.parse(path, format=attempt_fmt)
                fmt = attempt_fmt
                break
            except Exception:
                g = rdflib.Graph()   # reset after failed parse attempt
        if fmt is None:
            raise ValueError(
                f"Could not parse {path!r} as turtle or xml. "
                "Please convert it to a supported format."
            )
    else:
        log.info("Parsing ontology with rdflib (%s): %s", fmt, path)
        g.parse(path, format=fmt)
 
    log.info("Graph loaded: %d triples", len(g))
 
    SUBCLASS  = RDFS.subClassOf
    RDFS_LBL  = RDFS.label
    RDFS_CMT  = RDFS.comment
    RDFS_DOM  = RDFS.domain   # property → domain class
    RDFS_RNG  = RDFS.range    # property → range class
 
    # collect all class URIs
    class_uris: set[str] = set()
    for s in g.subjects(RDF.type, OWL.Class):
        if isinstance(s, URIRef):
            class_uris.add(str(s))
    for s in g.subjects(RDF.type, RDFS.Class):
        if isinstance(s, URIRef):
            class_uris.add(str(s))
    for s, o in g.subject_objects(SUBCLASS):
        if isinstance(s, URIRef):
            class_uris.add(str(s))
        if isinstance(o, URIRef):
            class_uris.add(str(o))
 
    log.info("Found %d class URIs", len(class_uris))
 
    # build Concept objects
    concepts:  Dict[str, Concept] = {}
    n_fallback = n_with_cmt = n_with_syn = 0
 
    for uri in class_uris:
        ref   = URIRef(uri)
        label = _pick_label(g, ref, RDFS_LBL, Literal, uri)
 
        if label == _uri_local_name(uri):
            n_fallback += 1
 
        comment  = _pick_comment(g, ref, RDFS_CMT, Literal)
        aliases = _pick_synonyms(g, ref, RDFS_LBL, Literal, label)
 
        if comment:
            n_with_cmt += 1
        if aliases:
            n_with_syn += 1
 
        concepts[uri] = Concept(
            uri      = uri,
            label    = label,
            comment  = comment,
            synonyms = aliases,
        )
 
    log.info(
        "%d/%d concepts: label-fallback=%d  with-comment=%d  with-synonyms=%d",
        len(concepts), len(class_uris),
        n_fallback, n_with_cmt, n_with_syn,
    )
 
    # collect subClassOf edges
    triplets: List[Tuple[str, str, str]] = []
 
    for s, o in sorted(g.subject_objects(SUBCLASS), key=lambda p: (str(p[0]), str(p[1]))):
        # sorted() gives deterministic parent selection for multi-parent classes
        if not (isinstance(s, URIRef) and isinstance(o, URIRef)):
            continue
        head, tail = str(s), str(o)
        if head == tail or head not in concepts or tail not in concepts:
            continue
        if tail not in concepts[head].parents:
            concepts[head].parents.append(tail)
            concepts[tail].children.append(head)
            triplets.append((head, "subClassOf", tail))
 
    # resolve parent_label
    for uri, concept in concepts.items():
        if concept.parents:
            parent_uri            = concept.parents[0]
            concept.parent_label  = concepts[parent_uri].label
 
    # build property → domain/range index
    #  Goal: for each concept C, find
    #    domain_label  → "what entity type commonly uses a property of type C"
    #    range_label   → "what value type does a property of type C yield"

 
    from collections import defaultdict, Counter
 
    # range_to_domain_votes[range_uri][domain_uri] = count
    range_to_domain_votes: Dict[str, Counter] = defaultdict(Counter)
    # domain_to_range_votes[domain_uri][range_uri] = count
    domain_to_range_votes: Dict[str, Counter] = defaultdict(Counter)
 
    # Collect all known property types
    property_types = (
        OWL.ObjectProperty,
        OWL.DatatypeProperty,
        OWL.AnnotationProperty,
        RDF.Property,
    )
 
    # Gather all property URIs
    prop_uris: set[URIRef] = set()
    for ptype in property_types:
        for p in g.subjects(RDF.type, ptype):
            if isinstance(p, URIRef):
                prop_uris.add(p)
    # Also collect anything that has rdfs:domain or rdfs:range triples
    for p in g.subjects(RDFS_DOM, None):
        if isinstance(p, URIRef):
            prop_uris.add(p)
    for p in g.subjects(RDFS_RNG, None):
        if isinstance(p, URIRef):
            prop_uris.add(p)
 
    n_props_indexed = 0
    for prop in prop_uris:
        domains = [str(o) for o in g.objects(prop, RDFS_DOM) if isinstance(o, URIRef)]
        ranges  = [str(o) for o in g.objects(prop, RDFS_RNG) if isinstance(o, URIRef)]
 
        # Filter to only known concept URIs
        domains = [d for d in domains if d in concepts]
        ranges  = [r for r in ranges  if r in concepts]
 
        if not (domains and ranges):
            continue
        n_props_indexed += 1
 
        for d_uri in domains:
            for r_uri in ranges:
                domain_to_range_votes[d_uri][r_uri] += 1
                range_to_domain_votes[r_uri][d_uri] += 1
 
    log.info(
        "Property index built: %d properties with resolvable domain+range",
        n_props_indexed,
    )
 
    # resolve domain_label and range_label per concept
    n_with_domain = n_with_range = 0
 
    for uri, concept in concepts.items():
        # domain_label: "what kind of entity has a property whose range = this concept"
        if uri in range_to_domain_votes and range_to_domain_votes[uri]:
            best_domain_uri       = range_to_domain_votes[uri].most_common(1)[0][0]
            concept.domain_label  = concepts[best_domain_uri].label
            n_with_domain        += 1
 
        # range_label: "what kind of value does a property yield when domain = this concept"
        if uri in domain_to_range_votes and domain_to_range_votes[uri]:
            best_range_uri       = domain_to_range_votes[uri].most_common(1)[0][0]
            concept.range_label  = concepts[best_range_uri].label
            n_with_range        += 1
 
    log.info(
        "TypeIndex enrichment: parent_label=%d  domain_label=%d  range_label=%d",
        sum(1 for c in concepts.values() if c.parent_label),
        n_with_domain,
        n_with_range,
    )

    for uri, concept in concepts.items():
        child_labels: List[str] = []
        for child_uri in concept.children:
            child = concepts.get(child_uri)
            if child and child.label:
                child_labels.append(child.label)
        concept.children_labels = child_labels[:10]   # cap at 10 subtypes
 
    # ── build ancestor_chain (parent → grandparent → ... → root labels) ───────
    # Provides the "breadcrumb" context for retrieval:
    # "BasketballPlayer → Athlete → Person → Agent"
    for uri, concept in concepts.items():
        chain:   List[str] = []
        visited: set       = {uri}
        current = concept.parents[0] if concept.parents else None
        while current and current not in visited:
            ancestor = concepts.get(current)
            if ancestor and ancestor.label:
                chain.append(ancestor.label)
            visited.add(current)
            parents = concepts[current].parents if current in concepts else []
            current = parents[0] if parents else None
        concept.ancestor_chain = chain   # [parent, grandparent, ..., root]
 
    n_with_children  = sum(1 for c in concepts.values() if c.children_labels)
    n_with_ancestors = sum(1 for c in concepts.values() if c.ancestor_chain)
    log.info(
        "Hierarchy enrichment — children=%d  ancestor_chain=%d",
        n_with_children, n_with_ancestors,
    )

    # ## enrich synonym
    # n_camel = 0

    # for uri, concept in concepts.items():
    #     existing = set(concept.aliases) | {concept.label}
    #     added_any = False
 
    #     # 1. CamelCase split from URI local name (more reliable than label)
    #     local = _uri_local_name(uri)
    #     for variant in _camel_split(local):
    #         if variant not in existing:
    #             concept.aliases.append(variant)
    #             existing.add(variant)
    #             added_any = True
    #     if added_any:
    #         n_camel += 1
 
    # log.info(
    #     "Alias enrichment — camelCase=%d URIs enriched",
    #     n_camel
    # )

    ## enrich descriptions from DBpedia SPARQL endpoint, only fills in when rdfs:comment is absent
    if abstracts:
        n_abstract = 0
        for uri, concept in concepts.items():
            if not concept.description and uri in abstracts:
                concept.description = abstracts[uri]
                n_abstract += 1
        log.info(
            "Abstract enrichment: %d concepts got description from dbo:abstract",
            n_abstract,
        )

 
    return OntologyCorpus(concepts=concepts, triplets=triplets)
 

def save_corpus(corpus: OntologyCorpus, path: str) -> None:
    data = {
        "concepts": {
            uri: {
                "concept_id":      c.concept_id,
                "label":    c.label,
                "aliases":  c.aliases,
                "description": c.description,
                "parents":  c.parent_label,
                "domain":   c.domain_label,
                "range":    c.range_label,
                "children":       c.children_labels,   # direct subclass labels
                "ancestor_chain": c.ancestor_chain,    # [parent, grandparent, ...]
            }
            for uri, c in corpus.concepts.items()
        },
        "triplets": corpus.triplets,
        "paths":    corpus.paths,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    n_desc    = sum(1 for c in corpus.concepts.values() if c.description)
    n_aliases = sum(1 for c in corpus.concepts.values() if c.aliases)
    n_parent  = sum(1 for c in corpus.concepts.values() if c.parent_label)
    n_children = sum(1 for c in corpus.concepts.values() if c.children_labels)
    n_anc      = sum(1 for c in corpus.concepts.values() if c.ancestor_chain)
    log.info(
        "Saved → %s  (%d concepts | desc=%d  aliases=%d  parent=%d)",
        path, len(corpus.concepts), n_desc, n_aliases, n_parent,
    )

def main():
    ont_paths = ONTO_PATH
    ont_names = ONTO_NAME or [Path(p).stem for p in ont_paths]
    out_dir   = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    # mined_headers = mine_headers_from_train(TRAIN_PATH, topk=TRAIN_HEADER_TOPK)
    for owl_path, name in zip(ont_paths, ont_names):
        cache = str(out_dir / f"{name}_cache.json")
        corpus = load_ontology(owl_path)

        if USE_WIKIPEDIA_API:
            # only fetch for concepts that still have no description
            missing_uris = [
                uri for uri, concept in corpus.concepts.items()
                if not concept.description
            ]
            all_uris_for_redirects = list(corpus.concepts.keys())
            fetch_uris = (
                all_uris_for_redirects if USE_WIKIPEDIA_REDIRECTS
                else missing_uris
            )

            log.info(
                "%d / %d concepts missing description — fetching from Wikipedia",
                len(missing_uris), len(corpus.concepts),
            )
            # if missing_uris:
            #     wiki_descriptions = fetch_wikipedia_descriptions(
            #         uris      = missing_uris,
            #         sleep     = WIKIPEDIA_API_SLEEP,
            #         max_chars = WIKIPEDIA_MAX_CHARS,
            #     )
            #     n_filled = 0
            #     for uri, desc in wiki_descriptions.items():
            #         corpus.concepts[uri].description = desc
            #         n_filled += 1
            #     log.info(
            #         "Wikipedia enrichment: %d / %d missing concepts filled",
            #         n_filled, len(missing_uris),
            #     )
    for owl_path, name in zip(ont_paths, ont_names):
        cache = str(out_dir / f"{name}_cache.json")
        log.info("── Processing %s → %s", owl_path, cache)

 
        # enrich descriptions + redirect aliases from Wikipedia API
        if USE_WIKIPEDIA_API:
            # fetch descriptions for concepts that have none
            # fetch redirects for ALL concepts (new aliases even if desc exists)
            missing_desc_uris = [
                uri for uri, concept in corpus.concepts.items()
                if not concept.description
            ]
            all_uris_for_redirects = list(corpus.concepts.keys())
 
            # if redirects enabled, fetch for all; else only fetch missing descs
            fetch_uris = (
                all_uris_for_redirects if USE_WIKIPEDIA_REDIRECTS
                else missing_desc_uris
            )
 
            log.info(
                "%d concepts missing description | fetching Wikipedia for %d URIs "
                "(redirects=%s)",
                len(missing_desc_uris), len(fetch_uris), USE_WIKIPEDIA_REDIRECTS,
            )
 
            if fetch_uris:
                wiki_results = fetch_wikipedia_descriptions(
                    uris             = fetch_uris,
                    sleep            = WIKIPEDIA_API_SLEEP,
                    max_chars        = WIKIPEDIA_MAX_CHARS,
                    fetch_redirects  = USE_WIKIPEDIA_REDIRECTS,
                    redirects_topk   = WIKIPEDIA_REDIRECTS_TOPK,
                )
 
                n_desc_filled = 0
                n_alias_added = 0
                for uri, result in wiki_results.items():
                    concept = corpus.concepts[uri]
 
                    # fill description only if still missing
                    if not concept.description and result["description"]:
                        concept.description = result["description"]
                        n_desc_filled += 1
 
                    # add redirect aliases (skip duplicates)
                    existing = set(concept.aliases) | {concept.label}
                    for alias in result["aliases"]:
                        if alias.lower() not in {e.lower() for e in existing}:
                            concept.aliases.append(alias)
                            existing.add(alias)
                            n_alias_added += 1
 
                log.info(
                    "Wikipedia enrichment: desc_filled=%d  aliases_added=%d",
                    n_desc_filled, n_alias_added,
                )
        save_corpus(corpus, cache)


if __name__=="__main__":
    main()



