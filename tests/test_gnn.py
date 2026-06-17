"""Tests for GNN training pipeline."""

import pytest

torch = pytest.importorskip("torch")
torch_geometric = pytest.importorskip("torch_geometric")

from mint_atlas.data.synthetic import generate_synthetic_mint
from mint_atlas.graph.builder import assign_chronological_sequence
from mint_atlas.inference.pipeline import TrainConfig, evaluate_model, train_model


def test_train_and_evaluate():
    graph, ground_truth = generate_synthetic_mint(
        num_obverse_dies=6,
        num_reverse_dies=4,
        seed=99,
    )
    assign_chronological_sequence(graph, ground_truth)
    config = TrainConfig(epochs=5, seed=99)
    model, history = train_model(graph, ground_truth, config)
    assert len(history) == 5
    report = evaluate_model(model, graph, ground_truth)
    assert report.num_dies == graph.num_dies
    assert -1.0 <= report.gnn_kendall_tau <= 1.0
