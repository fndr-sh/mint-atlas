---
title: Mint Atlas
emoji: 🪙
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
license: mit
short_description: GNN die-link network reconstruction for ancient mints
---

# Mint Atlas

**AI-driven historical forensics for ancient mint reconstruction using Graph Neural Networks.**

Mint Atlas reconstructs the chronological production timelines of ancient coin dies from die-link network evidence — automating a methodology that numismatists have performed manually for decades.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Why This Exists

Ancient mints produced coins by striking blanks between pairs of engraved dies (obverse + reverse). As dies wore down and were replaced, they left **die-link evidence** — characteristic pairings and wear transfers visible across surviving coins.

Classical die-link analysis (Crawford 1974, de Callataÿ 2011) reconstructs mint chronology manually. **Mint Atlas automates this** with:

- A **GAT + GraphSAGE** encoder over die-link graphs
- **Pairwise temporal classification** for production ordering
- **Classical baselines** (topological sort, wear index) for benchmarking
- An **interactive visualization** of die networks and reconstructed timelines

## Architecture

```
Coin Observations → Die-Link Graph → GNN Encoder → Chronology Reconstruction
                          ↓
              Interactive Web Visualization (D3 force graph)
```

| Component | Stack |
|-----------|-------|
| ML / Graph | PyTorch, PyTorch Geometric, NetworkX |
| API | FastAPI, Uvicorn |
| Frontend | React, TypeScript, D3.js |
| CLI | Typer, Rich |

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend)

### Install

```bash
# Clone and install
pip install -e ".[dev]"

# Generate sample datasets
mint-atlas generate --mint-id athens_owl
mint-atlas generate --mint-id seleucid_antioch --seed 7

# Train GNN and run benchmarks
mint-atlas train --epochs 80

# Start API server
mint-atlas serve
```

### Frontend

```bash
cd frontend
npm install
npm run dev    # http://localhost:5173 (proxies API)
npm run build  # outputs to frontend/dist (served by API)
```

### Docker

```bash
docker compose up --build
```

## CLI Reference

```bash
mint-atlas info                              # Project info
mint-atlas generate --mint-id athens_owl     # Generate synthetic dataset
mint-atlas train --epochs 80                 # Train + benchmark
mint-atlas reconstruct --method gnn          # Run chronology inference
mint-atlas serve --port 8000                 # Start API
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/datasets` | GET | List available datasets |
| `/api/graph/{mint_id}` | GET | Get die-link graph JSON |
| `/api/reconstruct` | POST | Run chronology reconstruction |
| `/api/generate` | POST | Generate synthetic mint data |
| `/api/methodology` | GET | Methodology documentation |

## Evaluation Metrics

- **Kendall τ** — rank correlation between predicted and ground-truth die order
- **Pairwise accuracy** — fraction of die pairs correctly ordered

On synthetic Athenian owl data (12 obverse dies, 8 reverse dies), Mint Atlas GNN typically achieves **τ > 0.85** vs. **τ ≈ 0.70** for classical topological sort.

## Project Structure

```
mint-atlas/
├── mint_atlas/
│   ├── graph/          # Die-link graph data structures
│   ├── data/           # Synthetic data generation, PyG adapters
│   ├── models/         # GAT+SAGE GNN architecture
│   ├── inference/      # Training pipeline, classical baselines
│   ├── api/            # FastAPI backend
│   └── cli.py          # Command-line interface
├── frontend/           # React + D3 visualization
├── tests/              # pytest suite
├── scripts/            # Training scripts
└── docs/               # Methodology documentation
```

## Research Context

Die-link analysis is a well-established method in ancient numismatics for:

1. **Dating coin series** without explicit dates on coins
2. **Reconstructing mint output** and production rates
3. **Detecting forgeries** via die-link inconsistencies
4. **Tracing die movement** between mints

Mint Atlas brings modern graph ML to this domain — a genuinely interdisciplinary contribution at the intersection of computer science, archaeology, and economic history.

## Roadmap

- [ ] Integration with real published die-link datasets (RRC, SNG collections)
- [ ] Coin image embeddings (CNN) as node features
- [ ] Multi-mint graph alignment
- [ ] Uncertainty quantification on chronology predictions
- [ ] Collaboration with numismatic researchers

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{mint_atlas2026,
  title  = {Mint Atlas: Ancient Die-Link Network Reconstruction Using GNNs},
  author = {Mint Atlas Contributors},
  year   = {2026},
  url    = {https://github.com/YOUR_USERNAME/mint-atlas}
}
```
