"""Mint Atlas command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from mint_atlas.data.synthetic import export_graph_json, generate_synthetic_mint, load_sample_dataset
from mint_atlas.graph.builder import assign_chronological_sequence
from mint_atlas.inference.classical import classical_topological, classical_wear_ordering
from mint_atlas.inference.pipeline import TrainConfig, run_full_pipeline

app = typer.Typer(
    name="mint-atlas",
    help="Ancient mint die-link network reconstruction using GNNs",
    add_completion=False,
)
console = Console()


@app.command()
def generate(
    mint_id: str = typer.Option("athens_owl", help="Mint dataset identifier"),
    obverse: int = typer.Option(12, help="Number of obverse dies"),
    reverse: int = typer.Option(8, help="Number of reverse dies"),
    seed: int = typer.Option(42, help="Random seed"),
    output: Path = typer.Option(Path("data/samples"), help="Output directory"),
):
    """Generate a synthetic die-link dataset."""
    graph, order = generate_synthetic_mint(
        mint_id=mint_id,
        num_obverse_dies=obverse,
        num_reverse_dies=reverse,
        seed=seed,
    )
    assign_chronological_sequence(graph, order)
    out_path = output / f"{mint_id}.json"
    export_graph_json(graph, out_path)
    console.print(f"[green]Generated[/green] {graph.num_coins} coins, {graph.num_dies} dies -> {out_path}")


@app.command()
def train(
    epochs: int = typer.Option(80, help="Training epochs"),
    seed: int = typer.Option(42, help="Random seed"),
    checkpoint_dir: Path = typer.Option(Path("checkpoints"), help="Checkpoint directory"),
):
    """Train Mint Atlas GNN on synthetic data and evaluate against baselines."""
    console.print("[bold]Training Mint Atlas GNN...[/bold]")
    config = TrainConfig(epochs=epochs, seed=seed, checkpoint_dir=str(checkpoint_dir))
    _, report, graph = run_full_pipeline(config)

    table = Table(title="Chronology Reconstruction Benchmark")
    table.add_column("Method", style="cyan")
    table.add_column("Kendall tau", justify="right")
    table.add_column("Pairwise Acc", justify="right")

    table.add_row("GNN (Mint Atlas)", f"{report.gnn_kendall_tau:.3f}", f"{report.gnn_pairwise_acc:.1%}")
    table.add_row("Topological Sort", f"{report.topological_kendall_tau:.3f}", "—")
    table.add_row("Wear Index", f"{report.wear_kendall_tau:.3f}", "—")
    table.add_row("", "", "")
    table.add_row("Dies", str(report.num_dies), "")
    table.add_row("Coins", str(report.num_coins), "")

    console.print(table)
    console.print(f"[green]Checkpoint saved[/green] -> {checkpoint_dir / 'mint_atlas_gnn.pt'}")


@app.command()
def reconstruct(
    mint_id: str = typer.Option("athens_owl", help="Dataset to reconstruct"),
    method: str = typer.Option("gnn", help="Method: gnn, topological, wear"),
):
    """Reconstruct die chronology for a dataset."""
    graph = load_sample_dataset(mint_id)
    ground_truth = graph.chronological_order()

    if method == "topological":
        result = classical_topological(graph)
    elif method == "wear":
        result = classical_wear_ordering(graph)
    else:
        from mint_atlas.inference.pipeline import reconstruct_chronology

        result = reconstruct_chronology(graph)

    console.print(f"[bold]Method:[/bold] {result.method}")
    console.print(f"[bold]Predicted order:[/bold] {' -> '.join(result.predicted_order[:15])}")
    if ground_truth:
        console.print(f"[bold]Ground truth:[/bold]   {' -> '.join(ground_truth[:15])}")
        console.print(f"Kendall τ = {result.kendall_tau:.3f}, Pairwise acc = {result.pairwise_accuracy:.1%}")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host"),
    port: int = typer.Option(8000, help="Port"),
    reload: bool = typer.Option(False, help="Auto-reload"),
):
    """Start the Mint Atlas API server."""
    import uvicorn

    uvicorn.run("mint_atlas.api.main:app", host=host, port=port, reload=reload)


@app.command()
def info():
    """Show project info and available datasets."""
    console.print("[bold cyan]Mint Atlas[/bold cyan] — GNN Die-Link Network Reconstruction")
    console.print("Ancient mint production timelines from coin die-link evidence.\n")
    datasets = ["athens_owl", "seleucid_antioch"]
    for ds in datasets:
        try:
            g = load_sample_dataset(ds)
            console.print(f"  • {ds}: {g.num_dies} dies, {g.num_coins} coins")
        except Exception:
            console.print(f"  • {ds}: (not generated yet)")


if __name__ == "__main__":
    app()
