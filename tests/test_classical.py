"""Tests for classical baselines."""

from mint_atlas.data.synthetic import generate_synthetic_mint
from mint_atlas.inference.classical import (
    classical_topological,
    classical_wear_ordering,
    evaluate_chronology,
    kendall_tau_distance,
    pairwise_accuracy,
)


def test_kendall_tau_perfect():
    order = ["A", "B", "C", "D"]
    assert kendall_tau_distance(order, order) == 1.0


def test_kendall_tau_reversed():
    assert kendall_tau_distance(["A", "B", "C"], ["C", "B", "A"]) == -1.0


def test_pairwise_accuracy():
    assert pairwise_accuracy(["A", "B", "C"], ["A", "B", "C"]) == 1.0
    assert pairwise_accuracy(["C", "B", "A"], ["A", "B", "C"]) == 0.0


def test_classical_baselines():
    graph, truth = generate_synthetic_mint(num_obverse_dies=8, seed=7)
    topo = classical_topological(graph)
    wear = classical_wear_ordering(graph)
    assert len(topo.predicted_order) == graph.num_dies
    assert len(wear.predicted_order) == graph.num_dies
    eval_topo = evaluate_chronology(topo.predicted_order, truth, "topological")
    assert eval_topo.kendall_tau is not None
