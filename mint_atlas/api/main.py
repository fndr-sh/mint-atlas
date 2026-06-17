"""FastAPI backend for Mint Atlas."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from mint_atlas.data.synthetic import export_graph_json, generate_synthetic_mint, load_sample_dataset
from mint_atlas.graph.builder import assign_chronological_sequence
from mint_atlas.graph.die_link import CoinObservation
from mint_atlas.inference.classical import classical_topological, classical_wear_ordering, evaluate_chronology
from mint_atlas.inference.pipeline import reconstruct_chronology

app = FastAPI(
    title="Mint Atlas API",
    description="GNN-based ancient mint die-link network reconstruction",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReconstructRequest(BaseModel):
    mint_id: str = "athens_owl"
    method: str = Field(default="gnn", pattern="^(gnn|topological|wear)$")


class GenerateRequest(BaseModel):
    num_obverse_dies: int = Field(default=12, ge=4, le=40)
    num_reverse_dies: int = Field(default=8, ge=3, le=30)
    seed: int = 42


@app.get("/health")
def health():
    return {"status": "ok", "service": "mint-atlas"}


@app.get("/api/datasets")
def list_datasets():
    return {
        "datasets": [
            {
                "id": "athens_owl",
                "name": "Athenian Owl Tetradrachm (Synthetic)",
                "description": "Synthetic die-link network modeled on Athenian owl coinage",
            },
            {
                "id": "seleucid_antioch",
                "name": "Seleucid Antioch (Synthetic)",
                "description": "Synthetic Seleucid mint production network",
            },
        ]
    }


@app.get("/api/graph/{mint_id}")
def get_graph(mint_id: str):
    try:
        graph = load_sample_dataset(mint_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "mint_id": graph.mint_id,
        "nodes": [
            {
                "id": die_id,
                "side": node.side.value,
                "wear_index": round(node.wear_index, 4),
                "estimated_sequence": node.estimated_sequence,
            }
            for die_id, node in graph.nodes.items()
        ],
        "edges": [
            {"source": e.source_die, "target": e.target_die, "coin_count": e.coin_count}
            for e in graph.coin_links
        ],
        "wear_transfers": [
            {"source": e.earlier_die, "target": e.later_die, "strength": round(e.transfer_strength, 4)}
            for e in graph.wear_transfers
        ],
        "stats": {
            "num_dies": graph.num_dies,
            "num_coins": graph.num_coins,
            "num_links": len(graph.coin_links),
        },
    }


@app.post("/api/generate")
def generate_mint(req: GenerateRequest):
    graph, order = generate_synthetic_mint(
        mint_id=f"synthetic_{req.seed}",
        num_obverse_dies=req.num_obverse_dies,
        num_reverse_dies=req.num_reverse_dies,
        seed=req.seed,
    )
    assign_chronological_sequence(graph, order)
    data_dir = Path("data/samples")
    export_graph_json(graph, data_dir / f"synthetic_{req.seed}.json")
    return {
        "mint_id": graph.mint_id,
        "ground_truth_order": order,
        "stats": {"num_dies": graph.num_dies, "num_coins": graph.num_coins},
    }


@app.post("/api/reconstruct")
def reconstruct(req: ReconstructRequest):
    graph = load_sample_dataset(req.mint_id)
    ground_truth = graph.chronological_order() or classical_topological(graph).predicted_order

    if req.method == "topological":
        result = classical_topological(graph)
    elif req.method == "wear":
        result = classical_wear_ordering(graph)
    else:
        result = reconstruct_chronology(graph)

    evaluated = evaluate_chronology(result.predicted_order, ground_truth, result.method)
    return {
        "method": evaluated.method,
        "predicted_order": evaluated.predicted_order,
        "metrics": {
            "kendall_tau": round(evaluated.kendall_tau or 0, 4),
            "pairwise_accuracy": round(evaluated.pairwise_accuracy or 0, 4),
        },
        "ground_truth_order": ground_truth,
    }


@app.get("/api/methodology")
def methodology():
    return {
        "problem": "Die-link analysis reconstructs the chronological order in which ancient coin dies were used.",
        "approach": "Mint Atlas models dies as nodes and coin pairings/wear transfers as edges in a graph.",
        "gnn_architecture": "GATConv + GraphSAGE encoder with pairwise temporal classifier and sequence ranker.",
        "baselines": ["wear_index", "topological_sort"],
        "metrics": ["kendall_tau", "pairwise_accuracy"],
        "references": [
            "Crawford, Roman Republican Coinage (die-link methodology)",
            "De Callataÿ, Quantitative die-studies in ancient numismatics",
        ],
    }


# Serve frontend build if present (must be last — catch-all)
_frontend = Path(__file__).parent.parent.parent / "frontend" / "dist"


@app.on_event("startup")
def _log_frontend_status():
    import logging
    logging.getLogger("mint_atlas").info(
        "Frontend dist %s", "found" if _frontend.exists() else "missing"
    )


if _frontend.exists():
    app.mount("/", StaticFiles(directory=str(_frontend), html=True), name="frontend")
