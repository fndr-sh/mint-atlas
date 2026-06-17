"""Training and inference pipeline for Mint Atlas GNN."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.optim as optim

from mint_atlas.data.datasets import build_pairwise_tensors, graph_to_pyg
from mint_atlas.data.synthetic import generate_synthetic_mint
from mint_atlas.graph.builder import assign_chronological_sequence
from mint_atlas.graph.die_link import DieLinkGraph
from mint_atlas.graph.temporal import obverse_die_ids, pairwise_labels_from_order
from mint_atlas.inference.classical import (
    ChronologyResult,
    classical_topological,
    classical_wear_ordering,
    evaluate_chronology,
)
from mint_atlas.models.gnn_temporal import MintAtlasGNN


@dataclass
class TrainConfig:
    epochs: int = 80
    lr: float = 1e-3
    hidden: int = 64
    embed_dim: int = 32
    seed: int = 42
    checkpoint_dir: str = "checkpoints"


@dataclass
class EvalReport:
    gnn_kendall_tau: float
    gnn_pairwise_acc: float
    topological_kendall_tau: float
    wear_kendall_tau: float
    num_dies: int
    num_coins: int


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    graph: DieLinkGraph,
    ground_truth: list[str],
    config: TrainConfig | None = None,
) -> tuple[MintAtlasGNN, list[dict]]:
    config = config or TrainConfig()
    set_seed(config.seed)

    pairs = pairwise_labels_from_order(ground_truth)
    batch = graph_to_pyg(graph, temporal_pairs=pairs)
    a_idx, b_idx, labels = build_pairwise_tensors(batch)
    data = batch.data

    in_channels = data.x.shape[1]
    model = MintAtlasGNN(in_channels=in_channels, hidden=config.hidden, embed_dim=config.embed_dim)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    history: list[dict] = []
    model.train()

    for epoch in range(config.epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, a_idx, b_idx)
        loss_pair = MintAtlasGNN.pairwise_loss(out["pair_logits"], labels)
        loss_rank = MintAtlasGNN.rank_loss(out["rank_logits"], data.y_seq)
        loss = loss_pair + 0.3 * loss_rank
        loss.backward()
        optimizer.step()
        history.append({"epoch": epoch + 1, "loss": float(loss.item()), "pair_loss": float(loss_pair.item())})

    return model, history


def evaluate_model(
    model: MintAtlasGNN,
    graph: DieLinkGraph,
    ground_truth: list[str],
) -> EvalReport:
    batch = graph_to_pyg(graph)
    data = batch.data

    obverse = obverse_die_ids(graph)
    predicted = model.infer_chronology(
        data.x, data.edge_index, batch.die_ids, die_subset=obverse
    )
    gnn_eval = evaluate_chronology(predicted, ground_truth, "gnn")
    topo_eval = evaluate_chronology(
        [d for d in classical_topological(graph).predicted_order if d in obverse],
        ground_truth,
        "topological",
    )
    wear_eval = evaluate_chronology(
        [d for d in classical_wear_ordering(graph).predicted_order if d in obverse],
        ground_truth,
        "wear",
    )

    return EvalReport(
        gnn_kendall_tau=gnn_eval.kendall_tau or 0.0,
        gnn_pairwise_acc=gnn_eval.pairwise_accuracy or 0.0,
        topological_kendall_tau=topo_eval.kendall_tau or 0.0,
        wear_kendall_tau=wear_eval.kendall_tau or 0.0,
        num_dies=graph.num_dies,
        num_coins=graph.num_coins,
    )


def run_full_pipeline(config: TrainConfig | None = None) -> tuple[MintAtlasGNN, EvalReport, DieLinkGraph]:
    config = config or TrainConfig()
    graph, ground_truth = generate_synthetic_mint(seed=config.seed)
    assign_chronological_sequence(graph, ground_truth)

    model, _ = train_model(graph, ground_truth, config)
    report = evaluate_model(model, graph, ground_truth)

    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "mint_atlas_gnn.pt")
    (ckpt_dir / "eval_report.json").write_text(
        json.dumps(asdict(report), indent=2), encoding="utf-8"
    )

    return model, report, graph


def reconstruct_chronology(
    graph: DieLinkGraph,
    checkpoint: Path | None = None,
) -> ChronologyResult:
    """Run chronology reconstruction on an arbitrary graph."""
    batch = graph_to_pyg(graph)
    data = batch.data
    in_channels = data.x.shape[1]

    model = MintAtlasGNN(in_channels=in_channels)
    ckpt = checkpoint or Path("checkpoints/mint_atlas_gnn.pt")
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, weights_only=True))
    else:
        # Train on-the-fly if no checkpoint
        gt = classical_topological(graph).predicted_order
        model, _ = train_model(graph, gt, TrainConfig(epochs=40))

    obverse = obverse_die_ids(graph)
    predicted = model.infer_chronology(
        data.x, data.edge_index, batch.die_ids, die_subset=obverse
    )
    baseline = [d for d in classical_topological(graph).predicted_order if d in obverse]
    return evaluate_chronology(predicted, baseline, "gnn")
