#!/usr/bin/env python3
"""Train Mint Atlas GNN and export evaluation report."""

from mint_atlas.inference.pipeline import TrainConfig, run_full_pipeline


def main():
    config = TrainConfig(epochs=80, seed=42)
    _, report, _ = run_full_pipeline(config)
    print("Training complete.")
    print(f"  GNN Kendall τ:        {report.gnn_kendall_tau:.4f}")
    print(f"  GNN Pairwise Acc:     {report.gnn_pairwise_acc:.2%}")
    print(f"  Topological Kendall τ: {report.topological_kendall_tau:.4f}")
    print(f"  Wear Index Kendall τ:  {report.wear_kendall_tau:.4f}")


if __name__ == "__main__":
    main()
