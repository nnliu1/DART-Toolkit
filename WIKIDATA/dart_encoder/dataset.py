from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Dict, List, Optional
import logging

try:
    from torch.utils.data import Dataset
except (ImportError, OSError):
    class Dataset:  # Allows CPU-only schema validation without a Torch runtime.
        pass

from .data_types import OntologyType, TrainSample
from .input_format import format_query, format_type
from .schema_v2 import iter_jsonl_records, normalize_ignored_qids, normalize_positive_qids

logger = logging.getLogger(__name__)

class CTADataset(Dataset):
    def __init__(
        self,
        train_path: str,
        ontology_path: str,
        max_cells: int = 10,
        max_parents: int = 3,
        seed: int = 42,
        mask_header_prob: float = 0.0,
        random_cell_sample: bool = False,
        min_majority_ratio: float = 0.0,  # filter samples below this ratio
        min_type_count: int = 1,          # filter types with fewer samples
        # Hard negative override
        hard_neg_path: Optional[str] = None,
    ):
        self.max_cells         = max_cells
        self.max_parents       = max_parents
        self.rng               = random.Random(seed)
        self.mask_header_prob  = mask_header_prob
        self.random_cell_sample= random_cell_sample
        self.training          = True   # set False at eval time

        # Load ontology
        self.ontology: Dict[str, OntologyType] = {}
        with open(ontology_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                # Fix double-quoted labels from label_db serialization
                for field in ["label", "description"]:
                    v = d.get(field, "")
                    if isinstance(v, str) and v.startswith('"') and v.endswith('"'):
                        d[field] = v[1:-1]
                self.ontology[d["qid"]] = OntologyType(
                    qid=d["qid"],
                    label=d.get("label", ""),
                    description=d.get("description", ""),
                    parents=d.get("parents", []),
                )

        # Load mined hard negatives─
        # Format: {"table_id_col": [qid1, qid2, ...], ...}
        # Key is f"{table_id}___{col_index}"
        mined_negs: Dict[str, List[str]] = {}
        if hard_neg_path and Path(hard_neg_path).exists():
            with open(hard_neg_path, encoding="utf-8") as f:
                mined_negs = json.load(f)
            logger.info(
                "Loaded mined hard negatives for %d samples", len(mined_negs)
            )

        # Load training samples
        raw_samples = []
        missing_types = set()
        skipped_missing_type_samples = 0
        for d in iter_jsonl_records(train_path):
            positive_qids = [
                qid for qid in normalize_positive_qids(d)
                if qid in self.ontology and self.ontology[qid].label
            ]
            if not positive_qids:
                missing_types.update(normalize_positive_qids(d))
                skipped_missing_type_samples += 1
                continue
            if d.get("majority_ratio", 1.0) < min_majority_ratio:
                continue
            d = dict(d)
            d["_valid_positive_qids"] = positive_qids
            d["_valid_primary_qid"] = (
                d["positive_type_qid"]
                if d["positive_type_qid"] in positive_qids
                else positive_qids[0]
            )
            raw_samples.append(d)

        # Filter low-frequency types
        if min_type_count > 1:
            from collections import Counter
            type_counts = Counter(d["_valid_primary_qid"] for d in raw_samples)
            before = len(raw_samples)
            raw_samples = [
                d for d in raw_samples
                if type_counts[d["_valid_primary_qid"]] >= min_type_count
            ]
            logger.info(
                "Type count filter (min=%d): %d → %d samples",
                min_type_count, before, len(raw_samples),
            )

        # Build final sample list
        self.samples: List[TrainSample] = []
        n_mined = 0
        for d in raw_samples:
            positive_qids = d["_valid_positive_qids"]
            pos_qid = d["_valid_primary_qid"]
            excluded_qids = set(positive_qids) | set(normalize_ignored_qids(d))
            table_id = d.get("table_id", "")
            col_idx  = d.get("col_index", -1)
            key      = f"{table_id}___{col_idx}"

            # Use mined hard negatives if available, else fall back to original
            if key in mined_negs:
                neg_qids = [
                    q for q in mined_negs[key]
                    if q in self.ontology and self.ontology[q].label
                    and q not in excluded_qids
                ]
                n_mined += 1
            else:
                neg_qids = [
                    q for q in d.get("hard_negative_type_qids", [])
                    if q in self.ontology and q not in excluded_qids
                ]
            neg_qids = list(dict.fromkeys(neg_qids))

            self.samples.append(TrainSample(
                anchor_header=d["anchor_header"],
                anchor_cells=d["anchor_cells"],
                positive_qid=pos_qid,
                hard_negative_qids=neg_qids,
                positive_qids=positive_qids,
            ))

        if missing_types:
            logger.warning(
                "Skipped %d samples (%d distinct QIDs): no positive type in ontology",
                skipped_missing_type_samples, len(missing_types),
            )
        logger.info(
            "Loaded %d training samples  "
            "(%d with mined hard negatives, %d with original)",
            len(self.samples), n_mined, len(self.samples) - n_mined,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        # Header mask augmentation
        header = s.anchor_header
        if self.training and self.mask_header_prob > 0:
            if self.rng.random() < self.mask_header_prob:
                header = ""

        # Random cell sample augmentation
        cells = s.anchor_cells
        if self.training and self.random_cell_sample and len(cells) > 1:
            # Randomly sample up to max_cells, preserving at least 1
            k = self.rng.randint(1, len(cells))
            cells = self.rng.sample(cells, k)

        query_text = format_query(header, cells, self.max_cells)
        pos_qids = s.all_positive_qids()
        pos_texts = [
            format_type(self.ontology[qid], self.max_parents)
            for qid in pos_qids
        ]
        neg_texts  = [
            format_type(self.ontology[q], self.max_parents)
            for q in s.hard_negative_qids
        ]
        return {
            "query_text": query_text,
            "pos_text": pos_texts[0],
            "pos_texts": pos_texts,
            "pos_qids": pos_qids,
            "neg_texts":  neg_texts,
            "neg_qids": s.hard_negative_qids,
        }


class CTACollator:
    def __init__(self, tokenizer, max_length: int = 256):
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[dict]) -> dict:
        query_texts = [b["query_text"] for b in batch]
        pos_texts_flat: List[str] = []
        pos_qids_flat: List[str] = []
        positive_qid_sets: List[List[str]] = []
        pos_counts: List[int] = []
        for item in batch:
            texts = item.get("pos_texts") or [item["pos_text"]]
            qids = item.get("pos_qids") or []
            pos_texts_flat.extend(texts)
            pos_qids_flat.extend(qids)
            positive_qid_sets.append(qids)
            pos_counts.append(len(texts))

        # Collect all hard negatives; record how many each sample has
        neg_texts_flat: List[str] = []
        neg_counts: List[int] = []
        for b in batch:
            neg_counts.append(len(b["neg_texts"]))
            neg_texts_flat.extend(b["neg_texts"])

        def tokenize(texts):
            return self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

        query_enc = tokenize(query_texts)
        pos_enc   = tokenize(pos_texts_flat)
        neg_enc   = tokenize(neg_texts_flat) if neg_texts_flat else None

        return {
            "query_enc": query_enc,
            "pos_enc":   pos_enc,
            "pos_counts": pos_counts,
            "positive_qid_sets": positive_qid_sets,
            "positive_qids_flat": pos_qids_flat,
            "neg_enc":   neg_enc,
            "neg_counts": neg_counts,
        }

