"""Tests for die-link graph construction."""

from mint_atlas.data.synthetic import generate_synthetic_mint
from mint_atlas.graph.builder import build_die_link_graph
from mint_atlas.graph.die_link import CoinObservation
from mint_atlas.graph.temporal import topological_chronology


def test_build_graph_from_observations():
    obs = [
        CoinObservation("C1", "O001", "R001", 17.0, 24.0),
        CoinObservation("C2", "O001", "R002", 16.9, 24.0),
        CoinObservation("C3", "O002", "R001", 16.5, 24.0),
    ]
    graph = build_die_link_graph(obs, mint_id="test")
    assert graph.num_dies == 4
    assert graph.num_coins == 3
    assert len(graph.coin_links) == 3


def test_synthetic_mint_generation():
    graph, order = generate_synthetic_mint(num_obverse_dies=6, num_reverse_dies=4, seed=0)
    assert graph.num_dies == 10
    assert graph.num_coins > 0
    assert len(order) == 6
    assert all(d.startswith("O") for d in order)


def test_topological_chronology():
    graph, order = generate_synthetic_mint(num_obverse_dies=8, seed=1)
    predicted = topological_chronology(graph)
    obverse_in_pred = [d for d in predicted if d.startswith("O")]
    obverse_in_truth = order
    # At least obverse dies should appear
    assert len(obverse_in_pred) >= len(obverse_in_truth) - 2
