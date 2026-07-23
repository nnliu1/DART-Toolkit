from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional



@dataclass
class OntologyType:
    qid: str
    label: str
    description: str
    parents: List[Dict]      # [{"qid": ..., "label": ...}, ...]



@dataclass
class TrainSample:
    anchor_header: str
    anchor_cells: List[str]
    positive_qid: str
    hard_negative_qids: List[str]
    positive_qids: Optional[List[str]] = None

    def all_positive_qids(self) -> List[str]:
        return self.positive_qids or [self.positive_qid]
