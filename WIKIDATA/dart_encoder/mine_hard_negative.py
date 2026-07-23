from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

from .data_types import OntologyType
from .input_format import format_query, format_type
from .mining_utils import (
    effective_retrieval_k,
    iter_mining_record_batches,
    sample_key,
    select_hard_negatives,
)
from .schema_v2 import iter_jsonl_records


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_ontology(ontology_path: str) -> Dict[str, OntologyType]:
    ontology: Dict[str, OntologyType] = {}
    with open(ontology_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            for field in ("label", "description"):
                value = data.get(field, "")
                if isinstance(value, str) and value.startswith('"') and value.endswith('"'):
                    data[field] = value[1:-1]
            ontology[data["qid"]] = OntologyType(
                qid=data["qid"],
                label=data.get("label", ""),
                description=data.get("description", ""),
                parents=data.get("parents", []),
            )
    return ontology


def mine_hard_negatives(
    model_path: str,
    train_path: str,
    ontology_path: str,
    output_path: str,
    top_k: int = 50,
    n_hard_negs: int = 5,
    min_rank: int = 2,
    max_rank: int = 50,
    min_score: float = 0.1,
    batch_size: int = 256,
    max_length: int = 256,
    max_cells: int = 10,
    max_parents: int = 3,
):
    import torch
    from transformers import AutoTokenizer
    from .model import BiEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    ontology = load_ontology(ontology_path)
    ontology_qids = set(ontology)
    logger.info("Loaded %d ontology types", len(ontology))

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = BiEncoder(model_path).to(device)
    model.eval()

    def encode(texts: List[str]) -> torch.Tensor:
        embeddings = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                encoded = tokenizer(
                    texts[start:start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                embeddings.append(model.encode(encoded).cpu())
        return torch.cat(embeddings, dim=0)

    all_qids = list(ontology)
    type_texts = [format_type(ontology[qid], max_parents) for qid in all_qids]
    logger.info("Encoding %d ontology types", len(type_texts))
    type_embs = encode(type_texts)

    mined: Dict[str, List[str]] = {}
    effective_k = effective_retrieval_k(top_k, max_rank, len(all_qids))
    processed = 0
    record_batches = iter_mining_record_batches(
        iter_jsonl_records(train_path), ontology_qids, batch_size
    )
    for batch in record_batches:
        query_texts = [
            format_query(record["anchor_header"], record["anchor_cells"], max_cells)
            for record in batch
        ]
        query_embs = encode(query_texts)
        batch_scores = query_embs @ type_embs.T
        values, indices = torch.topk(batch_scores, k=effective_k, dim=1)
        for row, record in enumerate(batch):
            key = sample_key(record)
            ranked_qids = [all_qids[index] for index in indices[row].tolist()]
            ranked_scores = values[row].tolist()
            mined[key] = select_hard_negatives(
                ranked_qids=ranked_qids,
                ranked_scores=ranked_scores,
                record=record,
                ontology_qids=ontology_qids,
                n_hard_negs=n_hard_negs,
                min_rank=min_rank,
                max_rank=max_rank,
                min_score=min_score,
            )
        processed += len(batch)
        logger.info("Mined %d samples", processed)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(mined, handle)
    logger.info("Saved hard negatives for %d samples to %s", len(mined), output)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Mine schema-v2 hard negatives")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_path", required=True,
                        help="JSONL file, shard directory, or glob")
    parser.add_argument("--ontology_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--n_hard_negs", type=int, default=5)
    parser.add_argument("--min_rank", type=int, default=2)
    parser.add_argument("--max_rank", type=int, default=50)
    parser.add_argument("--min_score", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_cells", type=int, default=10)
    parser.add_argument("--max_parents", type=int, default=3)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    mine_hard_negatives(**vars(args))
