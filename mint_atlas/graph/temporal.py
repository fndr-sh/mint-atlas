"""Temporal ordering algorithms for die-link chains."""

from __future__ import annotations

import networkx as nx

from mint_atlas.graph.die_link import DieLinkGraph, DieSide, WearTransferEdge


def obverse_die_ids(graph: DieLinkGraph) -> list[str]:
    """Return obverse die IDs sorted by ID for stable ordering."""
    return sorted(
        d for d in graph.die_ids()
        if graph.nodes[d].side == DieSide.OBVERSE or d.startswith("O")
    )


def pairwise_labels_from_order(order: list[str]) -> list[tuple[str, str, int]]:
    """Generate pairwise labels from a known chronological order."""
    labels: list[tuple[str, str, int]] = []
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            if i == j:
                continue
            labels.append((a, b, 1 if i < j else 0))
    return labels


def build_wear_dag(graph: DieLinkGraph) -> nx.DiGraph:
    """Build directed acyclic graph from wear transfer evidence."""
    dag = nx.DiGraph()
    for die_id in graph.nodes:
        dag.add_node(die_id, **{"wear": graph.nodes[die_id].wear_index})
    for edge in graph.wear_transfers:
        dag.add_edge(edge.earlier_die, edge.later_die, weight=edge.transfer_strength)
    return dag


def topological_chronology(graph: DieLinkGraph) -> list[str]:
    """
    Classical die-link chronology via topological sort on wear-transfer DAG.
    Falls back to wear-index ordering when DAG has cycles or is empty.
    """
    dag = build_wear_dag(graph)
    if dag.number_of_edges() == 0:
        return sorted(
            graph.die_ids(),
            key=lambda d: graph.nodes[d].wear_index,
        )

    try:
        return list(nx.topological_sort(dag))
    except nx.NetworkXUnfeasible:
        # Cycle detected — break ties by wear index
        condensed = nx.condensation(dag)
        comp_order = list(nx.topological_sort(condensed))
        components: dict[int, list[str]] = {c: [] for c in condensed.nodes}
        for node in dag.nodes:
            comp = condensed.graph["mapping"][node]
            components[comp].append(node)
        result: list[str] = []
        for comp in comp_order:
            members = sorted(components[comp], key=lambda d: graph.nodes[d].wear_index)
            result.extend(members)
        return result


def pairwise_temporal_labels(
    graph: DieLinkGraph,
    die_subset: list[str] | None = None,
) -> list[tuple[str, str, int]]:
    """
    Generate (die_a, die_b, label) triples for temporal ordering.
    label=1 if die_a precedes die_b, else 0.
    """
    order = topological_chronology(graph)
    if die_subset:
        order = [d for d in order if d in die_subset]
    rank = {die_id: i for i, die_id in enumerate(order)}
    labels: list[tuple[str, str, int]] = []
    dies = die_subset or graph.die_ids()
    for i, a in enumerate(dies):
        for b in dies[i + 1 :]:
            if a not in rank or b not in rank:
                continue
            if rank[a] < rank[b]:
                labels.append((a, b, 1))
            elif rank[a] > rank[b]:
                labels.append((a, b, 0))
    return labels


def infer_wear_transfers_from_sequence(
    graph: DieLinkGraph,
    true_order: list[str],
) -> list[WearTransferEdge]:
    """Infer wear transfer edges from known chronological order (for training data)."""
    rank = {die_id: i for i, die_id in enumerate(true_order)}
    transfers: list[WearTransferEdge] = []
    obverse_dies = [d for d in true_order if graph.nodes[d].side.value == "obverse" or d.startswith("O")]
    for i in range(len(obverse_dies) - 1):
        a, b = obverse_dies[i], obverse_dies[i + 1]
        if a in rank and b in rank:
            strength = 0.5 + 0.5 * (graph.nodes[b].wear_index - graph.nodes[a].wear_index)
            transfers.append(
                WearTransferEdge(
                    earlier_die=a,
                    later_die=b,
                    transfer_strength=float(max(0.1, min(1.0, strength))),
                )
            )
    return transfers
