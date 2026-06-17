"""Classical baselines for die-link chronology reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mint_atlas.graph.die_link import DieLinkGraph
from mint_atlas.graph.temporal import topological_chronology


@dataclass
class ChronologyResult:
    method: str
    predicted_order: list[str]
    kendall_tau: float | None = None
    pairwise_accuracy: float | None = None


def kendall_tau_distance(order_a: list[str], order_b: list[str]) -> float:
    """Compute Kendall tau correlation between two orderings."""
    common = [d for d in order_a if d in order_b]
    if len(common) < 2:
        return 0.0
    rank_a = {d: i for i, d in enumerate(order_a) if d in common}
    rank_b = {d: i for i, d in enumerate(order_b) if d in common}
    concordant = discordant = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            a, b = common[i], common[j]
            sign_a = np.sign(rank_a[b] - rank_a[a])
            sign_b = np.sign(rank_b[b] - rank_b[a])
            if sign_a == sign_b:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 0.0


def pairwise_accuracy(predicted: list[str], ground_truth: list[str]) -> float:
    """Fraction of die pairs correctly ordered."""
    common = [d for d in ground_truth if d in predicted]
    if len(common) < 2:
        return 0.0
    rank_p = {d: i for i, d in enumerate(predicted)}
    rank_g = {d: i for i, d in enumerate(ground_truth)}
    correct = total = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            a, b = common[i], common[j]
            if (rank_p[a] < rank_p[b]) == (rank_g[a] < rank_g[b]):
                correct += 1
            total += 1
    return correct / total if total else 0.0


def classical_wear_ordering(graph: DieLinkGraph) -> ChronologyResult:
    """Order dies by ascending wear index (naive baseline)."""
    order = sorted(graph.die_ids(), key=lambda d: graph.nodes[d].wear_index)
    return ChronologyResult(method="wear_index", predicted_order=order)


def classical_topological(graph: DieLinkGraph) -> ChronologyResult:
    """Order dies via wear-transfer DAG topological sort."""
    order = topological_chronology(graph)
    return ChronologyResult(method="topological_sort", predicted_order=order)


def evaluate_chronology(
    predicted: list[str],
    ground_truth: list[str],
    method: str,
) -> ChronologyResult:
    return ChronologyResult(
        method=method,
        predicted_order=predicted,
        kendall_tau=kendall_tau_distance(predicted, ground_truth),
        pairwise_accuracy=pairwise_accuracy(predicted, ground_truth),
    )
