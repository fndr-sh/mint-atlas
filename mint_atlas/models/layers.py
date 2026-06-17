"""GNN layers for die-link network analysis."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv


class DieLinkEncoder(nn.Module):
    """Shared GNN encoder producing die embeddings."""

    def __init__(self, in_channels: int, hidden: int = 64, out_channels: int = 32, heads: int = 4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden, heads=heads, concat=True, dropout=0.2)
        self.conv2 = SAGEConv(hidden * heads, out_channels)
        self.bn = nn.BatchNorm1d(hidden * heads)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = self.bn(h)
        h = F.elu(h)
        h = self.conv2(h, edge_index)
        return h


class TemporalPairClassifier(nn.Module):
    """Predict whether die_a precedes die_b using concatenated embeddings."""

    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2 + 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        emb: torch.Tensor,
        a_idx: torch.Tensor,
        b_idx: torch.Tensor,
    ) -> torch.Tensor:
        ea = emb[a_idx]
        eb = emb[b_idx]
        wear_diff = (ea[:, :1] - eb[:, :1]) if ea.shape[1] >= 1 else torch.zeros(len(a_idx), 1)
        side_diff = torch.zeros(len(a_idx), 1, device=emb.device)
        pair = torch.cat([ea, eb, wear_diff, side_diff], dim=-1)
        return self.mlp(pair).squeeze(-1)


class SequenceRanker(nn.Module):
    """Predict absolute sequence rank per die node."""

    def __init__(self, embed_dim: int = 32, max_ranks: int = 64):
        super().__init__()
        self.head = nn.Linear(embed_dim, max_ranks)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.head(emb)
