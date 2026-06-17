"""Synthetic ancient mint data generator for training and benchmarking."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from mint_atlas.graph.builder import build_die_link_graph
from mint_atlas.graph.die_link import CoinObservation, DieLinkGraph, WearTransferEdge
from mint_atlas.graph.temporal import infer_wear_transfers_from_sequence


def generate_synthetic_mint(
    mint_id: str = "athens_owl_synthetic",
    num_obverse_dies: int = 12,
    num_reverse_dies: int = 8,
    coins_per_pair_range: tuple[int, int] = (2, 15),
    style_dim: int = 8,
    seed: int = 42,
) -> tuple[DieLinkGraph, list[str]]:
    """
    Generate a synthetic die-link network mimicking ancient mint production.

    Obverse dies wear out sequentially; reverse dies are reused across obverses.
    Returns graph and ground-truth obverse die chronology.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    obverse_order = [f"O{i:03d}" for i in range(num_obverse_dies)]
    reverse_dies = [f"R{j:03d}" for j in range(num_reverse_dies)]

    obverse_styles = {d: np_rng.normal(0, 1, style_dim).tolist() for d in obverse_order}
    reverse_styles = {d: np_rng.normal(0, 1, style_dim).tolist() for d in reverse_dies}

    observations: list[CoinObservation] = []
    coin_idx = 0

    for o_rank, obv in enumerate(obverse_order):
        wear = o_rank / max(1, num_obverse_dies - 1)
        active_reverses = rng.sample(
            reverse_dies,
            k=rng.randint(max(2, num_reverse_dies // 3), num_reverse_dies),
        )
        for rev in active_reverses:
            n_coins = rng.randint(*coins_per_pair_range)
            for _ in range(n_coins):
                weight = 17.2 - wear * 1.5 + np_rng.normal(0, 0.1)
                style = [
                    (obverse_styles[obv][k] + reverse_styles[rev][k]) / 2
                    for k in range(style_dim)
                ]
                observations.append(
                    CoinObservation(
                        coin_id=f"C{coin_idx:05d}",
                        obverse_die_id=obv,
                        reverse_die_id=rev,
                        weight_g=round(float(weight), 2),
                        diameter_mm=round(24.0 + np_rng.normal(0, 0.3), 2),
                        find_site=rng.choice(["athens", "olympia", "delos", "smyrna"]),
                        hoard_id=rng.choice([None, "hoard_alpha", "hoard_beta"]),
                        style_vector=tuple(style),
                    )
                )
                coin_idx += 1

    graph = build_die_link_graph(observations, mint_id=mint_id)
    transfers = infer_wear_transfers_from_sequence(graph, obverse_order)
    graph.wear_transfers = transfers

    for o_rank, obv in enumerate(obverse_order):
        graph.nodes[obv].wear_index = o_rank / max(1, num_obverse_dies - 1)
        graph.nodes[obv].estimated_sequence = o_rank

    return graph, obverse_order


def export_graph_json(graph: DieLinkGraph, path: Path) -> None:
    """Serialize graph to JSON for API/frontend consumption."""
    data = {
        "mint_id": graph.mint_id,
        "nodes": [
            {
                "id": die_id,
                "side": node.side.value,
                "wear_index": node.wear_index,
                "estimated_sequence": node.estimated_sequence,
                "style_vector": node.style_vector.tolist(),
            }
            for die_id, node in graph.nodes.items()
        ],
        "edges": [
            {
                "source": e.source_die,
                "target": e.target_die,
                "coin_count": e.coin_count,
                "type": "coin_link",
            }
            for e in graph.coin_links
        ],
        "wear_transfers": [
            {
                "source": e.earlier_die,
                "target": e.later_die,
                "strength": e.transfer_strength,
            }
            for e in graph.wear_transfers
        ],
        "observations": [
            {
                "coin_id": o.coin_id,
                "obverse": o.obverse_die_id,
                "reverse": o.reverse_die_id,
                "weight_g": o.weight_g,
                "find_site": o.find_site,
            }
            for o in graph.observations[:200]
        ],
        "stats": {
            "num_dies": graph.num_dies,
            "num_coins": graph.num_coins,
            "num_links": len(graph.coin_links),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_sample_dataset(name: str = "athens_owl") -> DieLinkGraph:
    """Load or generate a named sample dataset."""
    data_dir = Path(__file__).parent.parent.parent / "data" / "samples"
    json_path = data_dir / f"{name}.json"

    if json_path.exists():
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        observations = [
            CoinObservation(
                coin_id=o["coin_id"],
                obverse_die_id=o["obverse"],
                reverse_die_id=o["reverse"],
                weight_g=o["weight_g"],
                diameter_mm=o.get("diameter_mm", 24.0),
                find_site=o.get("find_site", "unknown"),
            )
            for o in raw.get("observations_full", raw.get("observations", []))
        ]
        graph = build_die_link_graph(observations, mint_id=raw["mint_id"])
        for node_data in raw["nodes"]:
            if node_data["id"] in graph.nodes:
                graph.nodes[node_data["id"]].wear_index = node_data["wear_index"]
                graph.nodes[node_data["id"]].estimated_sequence = node_data.get("estimated_sequence")
        return graph

    graph, _ = generate_synthetic_mint(mint_id=name, seed=hash(name) % 10000)
    export_graph_json(graph, json_path)
    return graph
