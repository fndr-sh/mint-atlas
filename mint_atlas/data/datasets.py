"""PyTorch Geometric dataset adapters for die-link graphs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import Data

from mint_atlas.graph.die_link import DieLinkGraph
from mint_atlas.graph.temporal import pairwise_temporal_labels


@dataclass
class GraphBatch:
    """Container for a converted die-link graph."""

    data: Data
    die_ids: list[str]
    temporal_pairs: list[tuple[str, str, int]]


def graph_to_pyg(
    graph: DieLinkGraph,
    temporal_pairs: list[tuple[str, str, int]] | None = None,
) -> GraphBatch:
    """Convert DieLinkGraph to PyG Data object with node features and edges."""
    die_ids = graph.die_ids()
    id_map = {d: i for i, d in enumerate(die_ids)}

    x = torch.stack(
        [torch.from_numpy(graph.nodes[d].to_feature_vector()) for d in die_ids]
    )

    # Undirected coin-link edges between obverse-reverse pairs
    edge_index_list: list[list[int]] = [[], []]
    for link in graph.coin_links:
        if link.source_die in id_map and link.target_die in id_map:
            s, t = id_map[link.source_die], id_map[link.target_die]
            edge_index_list[0].extend([s, t])
            edge_index_list[1].extend([t, s])

    # Wear transfer directed edges
    for wt in graph.wear_transfers:
        if wt.earlier_die in id_map and wt.later_die in id_map:
            s, t = id_map[wt.earlier_die], id_map[wt.later_die]
            edge_index_list[0].append(s)
            edge_index_list[1].append(t)

    edge_index = torch.tensor(edge_index_list, dtype=torch.long)

    # Node-level sequence labels (for obverse dies with known order)
    y_seq = torch.full((len(die_ids),), -1, dtype=torch.long)
    for i, d in enumerate(die_ids):
        seq = graph.nodes[d].estimated_sequence
        if seq is not None:
            y_seq[i] = seq

    data = Data(x=x, edge_index=edge_index, y_seq=y_seq)
    pairs = temporal_pairs if temporal_pairs is not None else pairwise_temporal_labels(graph)

    return GraphBatch(data=data, die_ids=die_ids, temporal_pairs=pairs)


def build_pairwise_tensors(
    batch: GraphBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (node_a_idx, node_b_idx, label) tensors for temporal pairwise training."""
    id_map = {d: i for i, d in enumerate(batch.die_ids)}
    pairs = batch.temporal_pairs
    if not pairs:
        return (
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.float),
        )
    a_idx = torch.tensor([id_map[a] for a, _, _ in pairs], dtype=torch.long)
    b_idx = torch.tensor([id_map[b] for _, b, _ in pairs], dtype=torch.long)
    labels = torch.tensor([float(l) for _, _, l in pairs], dtype=torch.float)
    return a_idx, b_idx, labels
