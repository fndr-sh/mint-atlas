"""GNN model for die chronology reconstruction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mint_atlas.models.layers import DieLinkEncoder, SequenceRanker, TemporalPairClassifier


class MintAtlasGNN(nn.Module):
    """
    End-to-end GNN for ancient mint die-link chronology reconstruction.

    Combines a GAT+GraphSAGE encoder with:
    - Pairwise temporal ordering head (primary)
    - Sequence ranking head (auxiliary)
    """

    def __init__(
        self,
        in_channels: int = 10,
        hidden: int = 64,
        embed_dim: int = 32,
        max_ranks: int = 64,
    ):
        super().__init__()
        self.encoder = DieLinkEncoder(in_channels, hidden, embed_dim)
        self.pair_classifier = TemporalPairClassifier(embed_dim)
        self.ranker = SequenceRanker(embed_dim, max_ranks)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def predict_pairwise(
        self,
        emb: torch.Tensor,
        a_idx: torch.Tensor,
        b_idx: torch.Tensor,
    ) -> torch.Tensor:
        return self.pair_classifier(emb, a_idx, b_idx)

    def predict_ranks(self, emb: torch.Tensor) -> torch.Tensor:
        return self.ranker(emb)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        a_idx: torch.Tensor | None = None,
        b_idx: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        emb = self.encode(x, edge_index)
        out: dict[str, torch.Tensor] = {"embeddings": emb, "rank_logits": self.predict_ranks(emb)}
        if a_idx is not None and b_idx is not None and len(a_idx) > 0:
            out["pair_logits"] = self.predict_pairwise(emb, a_idx, b_idx)
        return out

    def infer_chronology(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        die_ids: list[str],
        die_subset: list[str] | None = None,
    ) -> list[str]:
        """Infer die chronology via pairwise tournament ranking."""
        self.eval()
        subset = die_subset or die_ids
        subset_idx = {die_ids.index(d) for d in subset if d in die_ids}
        index_list = sorted(subset_idx)

        with torch.no_grad():
            emb = self.encode(x, edge_index)
            n = len(index_list)
            scores = torch.zeros(n)
            for i in range(n):
                for j in range(i + 1, n):
                    logit = self.predict_pairwise(
                        emb,
                        torch.tensor([index_list[i]]),
                        torch.tensor([index_list[j]]),
                    ).item()
                    if logit > 0:
                        scores[i] += 1
                    else:
                        scores[j] += 1
            order_idx = scores.argsort(descending=True)
            return [die_ids[index_list[i]] for i in order_idx]

    @staticmethod
    def pairwise_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, labels)

    @staticmethod
    def rank_loss(logits: torch.Tensor, y_seq: torch.Tensor) -> torch.Tensor:
        mask = y_seq >= 0
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        return F.cross_entropy(logits[mask], y_seq[mask])
