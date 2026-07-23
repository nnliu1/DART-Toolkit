from __future__ import annotations
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .data_types import OntologyType, TrainSample
from .input_format import format_query, format_type
from .loss_utils import build_positive_mask_rows


def multi_positive_contrastive_loss(
    q_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    pos_counts: List[int],
    positive_qid_sets: List[List[str]],
    temperature,
    *,
    positive_qids_flat: Optional[List[str]] = None,
    neg_emb: Optional[torch.Tensor] = None,
    neg_counts: Optional[List[int]] = None,
) -> torch.Tensor:
    """Multi-positive InfoNCE plus query-specific explicit negatives."""
    scale = temperature.clamp(min=1e-4) if isinstance(
        temperature, torch.Tensor
    ) else max(float(temperature), 1e-4)
    logits = q_emb @ pos_emb.T / scale
    mask_rows = build_positive_mask_rows(
        pos_counts, positive_qid_sets, positive_qids_flat
    )
    positive_mask = torch.tensor(
        mask_rows, dtype=torch.bool, device=logits.device
    )
    if not positive_mask.any(dim=1).all():
        raise ValueError("Every query must have at least one positive")
    log_probabilities = F.log_softmax(logits, dim=1)
    positive_log_probabilities = log_probabilities.masked_fill(
        ~positive_mask, float("-inf")
    )
    loss_inbatch = -torch.logsumexp(
        positive_log_probabilities, dim=1
    ).mean()
    if neg_emb is None or not neg_counts or sum(neg_counts) == 0:
        return loss_inbatch
    pos_splits = torch.split(pos_emb, pos_counts, dim=0)
    neg_splits = torch.split(neg_emb, neg_counts, dim=0)
    hard_losses = []
    for query, positives, negatives in zip(q_emb, pos_splits, neg_splits):
        if negatives.shape[0] == 0:
            continue
        positive_scores = query.unsqueeze(0) @ positives.T / scale
        negative_scores = query.unsqueeze(0) @ negatives.T / scale
        candidate_log_probabilities = F.log_softmax(
            torch.cat([positive_scores, negative_scores], dim=1), dim=1
        )
        hard_losses.append(
            -torch.logsumexp(
                candidate_log_probabilities[:, :positives.shape[0]], dim=1
            ).squeeze(0)
        )
    return (
        loss_inbatch + torch.stack(hard_losses).mean()
        if hard_losses else loss_inbatch
    )


class BiEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.05,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def mean_pool(self, model_output, attention_mask) -> torch.Tensor:
        token_embeddings = model_output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def encode(self, encoding: dict) -> torch.Tensor:
        out = self.encoder(**encoding)
        emb = self.mean_pool(out, encoding["attention_mask"])
        return F.normalize(emb, dim=-1)

    def forward(
        self,
        query_enc: dict,
        pos_enc: dict,
        neg_enc: Optional[dict],
        neg_counts: List[int],
        pos_counts: Optional[List[int]] = None,
        positive_qid_sets: Optional[List[List[str]]] = None,
        positive_qids_flat: Optional[List[str]] = None,
    ) -> torch.Tensor:
        q_emb = self.encode(query_enc)
        pos_emb = self.encode(pos_enc)
        pos_counts = pos_counts or [1] * len(q_emb)
        positive_qid_sets = positive_qid_sets or [[] for _ in range(len(q_emb))]
        neg_emb = (
            self.encode(neg_enc)
            if neg_enc is not None and sum(neg_counts) > 0 else None
        )
        return multi_positive_contrastive_loss(
            q_emb, pos_emb, pos_counts, positive_qid_sets, self.temperature,
            positive_qids_flat=positive_qids_flat,
            neg_emb=neg_emb,
            neg_counts=neg_counts,
        )


@torch.no_grad()
def evaluate_recall(
    model: BiEncoder,
    val_samples: List[TrainSample],
    ontology: Dict[str, OntologyType],
    tokenizer,
    device: torch.device,
    batch_size: int = 64,
    max_length: int = 256,
    max_cells: int = 10,
    max_parents: int = 3,
    k_values: List[int] = [1, 5, 20],
) -> Dict[str, float]:
    model.eval()

    def tokenize(texts):
        return {
            key: value.to(device)
            for key, value in tokenizer(
                texts, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).items()
        }

    all_qids = list(ontology)
    all_texts = [format_type(ontology[qid], max_parents) for qid in all_qids]
    type_embs = []
    for start in range(0, len(all_texts), batch_size):
        type_embs.append(model.encode(tokenize(all_texts[start:start + batch_size])).cpu())
    type_embs = torch.cat(type_embs, dim=0)
    qid_to_idx = {qid: index for index, qid in enumerate(all_qids)}
    hits = {k: 0 for k in k_values}
    total = 0
    for start in range(0, len(val_samples), batch_size):
        batch = val_samples[start:start + batch_size]
        queries = [
            format_query(sample.anchor_header, sample.anchor_cells, max_cells)
            for sample in batch
        ]
        scores = model.encode(tokenize(queries)).cpu() @ type_embs.T
        for row, sample in enumerate(batch):
            positive_indices = {
                qid_to_idx[qid] for qid in sample.all_positive_qids()
                if qid in qid_to_idx
            }
            if not positive_indices:
                continue
            ranked = scores[row].argsort(descending=True).tolist()
            for k in k_values:
                if positive_indices.intersection(ranked[:k]):
                    hits[k] += 1
            total += 1
    model.train()
    return {f"Recall@{k}": hits[k] / max(total, 1) for k in k_values}
