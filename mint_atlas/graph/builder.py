"""Build die-link graphs from coin observation records."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from mint_atlas.graph.die_link import (
    CoinObservation,
    DieLinkEdge,
    DieLinkGraph,
    DieNode,
    DieSide,
    WearTransferEdge,
)


def _infer_side(die_id: str) -> DieSide:
    if die_id.startswith("O") or "_O" in die_id:
        return DieSide.OBVERSE
    if die_id.startswith("R") or "_R" in die_id:
        return DieSide.REVERSE
    return DieSide.OBVERSE


def build_die_link_graph(
    observations: list[CoinObservation],
    mint_id: str = "synthetic",
    wear_transfers: list[WearTransferEdge] | None = None,
) -> DieLinkGraph:
    """Construct a die-link graph from coin observations."""
    nodes: dict[str, DieNode] = {}
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    die_styles: dict[str, list[np.ndarray]] = defaultdict(list)
    die_wear: dict[str, list[float]] = defaultdict(list)

    for obs in observations:
        for die_id in obs.dies():
            if die_id not in nodes:
                style = np.array(obs.style_vector, dtype=np.float32) if obs.style_vector else np.zeros(8)
                nodes[die_id] = DieNode(
                    die_id=die_id,
                    side=_infer_side(die_id),
                    style_vector=style if len(style) == 8 else np.pad(style, (0, max(0, 8 - len(style))))[:8],
                    mint_id=mint_id,
                )
            if obs.style_vector:
                die_styles[die_id].append(np.array(obs.style_vector, dtype=np.float32)[:8])
        pair_counts[(obs.obverse_die_id, obs.reverse_die_id)] += 1
        wear_proxy = 1.0 - (obs.weight_g / 17.0)
        die_wear[obs.obverse_die_id].append(wear_proxy)
        die_wear[obs.reverse_die_id].append(wear_proxy)

    for die_id, node in nodes.items():
        if die_styles[die_id]:
            node.style_vector = np.mean(die_styles[die_id], axis=0)
        if die_wear[die_id]:
            node.wear_index = float(np.clip(np.mean(die_wear[die_id]), 0.0, 1.0))

    coin_links = [
        DieLinkEdge(source_die=o, target_die=r, coin_count=c)
        for (o, r), c in pair_counts.items()
    ]

    return DieLinkGraph(
        mint_id=mint_id,
        nodes=nodes,
        coin_links=coin_links,
        wear_transfers=wear_transfers or [],
        observations=observations,
    )


def assign_chronological_sequence(graph: DieLinkGraph, order: list[str]) -> DieLinkGraph:
    """Assign estimated production sequence indices to dies."""
    for rank, die_id in enumerate(order):
        if die_id in graph.nodes:
            graph.nodes[die_id].estimated_sequence = rank
    return graph
