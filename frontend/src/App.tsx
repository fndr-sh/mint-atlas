import { useCallback, useEffect, useState } from "react";
import { apiUrl } from "./api";
import GraphView from "./components/GraphView";

interface GraphData {
  mint_id: string;
  nodes: Array<{
    id: string;
    side: string;
    wear_index: number;
    estimated_sequence: number | null;
  }>;
  edges: Array<{ source: string; target: string; coin_count: number }>;
  wear_transfers: Array<{ source: string; target: string; strength: number }>;
  stats: { num_dies: number; num_coins: number; num_links: number };
}

interface ReconstructResult {
  method: string;
  predicted_order: string[];
  metrics: { kendall_tau: number; pairwise_accuracy: number };
  ground_truth_order: string[];
}

const DATASETS = [
  { id: "athens_owl", label: "Athenian Owl (Synthetic)" },
  { id: "seleucid_antioch", label: "Seleucid Antioch (Synthetic)" },
];

const METHODS = [
  { id: "gnn", label: "GNN (Mint Atlas)" },
  { id: "topological", label: "Topological Sort" },
  { id: "wear", label: "Wear Index" },
];

export default function App() {
  const [dataset, setDataset] = useState("athens_owl");
  const [method, setMethod] = useState("gnn");
  const [graph, setGraph] = useState<GraphData | null>(null);
  const [result, setResult] = useState<ReconstructResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadGraph = useCallback(async (mintId: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(apiUrl(`/api/graph/${mintId}`));
      if (!res.ok) throw new Error(`Failed to load graph: ${res.status}`);
      const data: GraphData = await res.json();
      setGraph(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, []);

  const runReconstruct = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(apiUrl("/api/reconstruct"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mint_id: dataset, method }),
      });
      if (!res.ok) throw new Error(`Reconstruction failed: ${res.status}`);
      const data: ReconstructResult = await res.json();
      setResult(data);
      await loadGraph(dataset);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, [dataset, method, loadGraph]);

  useEffect(() => {
    loadGraph(dataset);
  }, [dataset, loadGraph]);

  return (
    <div className="app">
      <header className="header">
        <div className="logo">
          <h1>Mint Atlas</h1>
          <span>die-link · GNN · chronology</span>
        </div>
        <div className="controls">
          <select value={dataset} onChange={(e) => setDataset(e.target.value)}>
            {DATASETS.map((d) => (
              <option key={d.id} value={d.id}>
                {d.label}
              </option>
            ))}
          </select>
          <select value={method} onChange={(e) => setMethod(e.target.value)}>
            {METHODS.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </select>
          <button className="primary" onClick={runReconstruct} disabled={loading}>
            {loading ? "Running…" : "Reconstruct Timeline"}
          </button>
        </div>
      </header>

      <main className="main">
        <div className="graph-panel">
          {loading && !graph ? (
            <div className="loading">Loading die-link network…</div>
          ) : graph ? (
            <GraphView
              nodes={graph.nodes}
              edges={graph.edges}
              wearTransfers={graph.wear_transfers}
              chronology={result?.predicted_order}
            />
          ) : (
            <div className="loading">{error ?? "No data"}</div>
          )}
        </div>

        <aside className="sidebar">
          {graph && (
            <div className="card">
              <h2>Network Stats</h2>
              <div className="stats-grid">
                <div className="stat">
                  <div className="value">{graph.stats.num_dies}</div>
                  <div className="label">Dies</div>
                </div>
                <div className="stat">
                  <div className="value">{graph.stats.num_coins}</div>
                  <div className="label">Coins</div>
                </div>
                <div className="stat">
                  <div className="value">{graph.stats.num_links}</div>
                  <div className="label">Die Links</div>
                </div>
                <div className="stat">
                  <div className="value">{graph.wear_transfers.length}</div>
                  <div className="label">Wear Transfers</div>
                </div>
              </div>
            </div>
          )}

          {result && (
            <div className="card">
              <h2>Reconstruction Metrics</h2>
              <div className="metrics">
                <div className="metric">
                  <div className="value">{result.metrics.kendall_tau.toFixed(3)}</div>
                  <div className="label">Kendall τ</div>
                </div>
                <div className="metric">
                  <div className="value">{(result.metrics.pairwise_accuracy * 100).toFixed(1)}%</div>
                  <div className="label">Pairwise Acc</div>
                </div>
              </div>
            </div>
          )}

          {result && (
            <div className="card">
              <h2>Predicted Chronology ({result.method})</h2>
              <div className="chronology-list">
                {result.predicted_order.slice(0, 20).map((dieId, i) => {
                  const node = graph?.nodes.find((n) => n.id === dieId);
                  return (
                    <div key={dieId} className="die">
                      <span className="rank">{i + 1}.</span>
                      <span className={node?.side === "obverse" ? "obverse" : "reverse"}>
                        {dieId}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          <div className="card">
            <h2>Legend</h2>
            <div className="legend">
              <div className="legend-item">
                <div className="legend-dot" style={{ background: "var(--obverse)" }} />
                Obverse die
              </div>
              <div className="legend-item">
                <div className="legend-dot" style={{ background: "var(--reverse)" }} />
                Reverse die
              </div>
            </div>
          </div>

          <div className="card methodology">
            <h2>Methodology</h2>
            <p>
              <strong>Die-link analysis</strong> reconstructs mint production order from shared
              die pairings and wear transfer evidence across ancient coins.
            </p>
            <p style={{ marginTop: "0.5rem" }}>
              Mint Atlas uses a <strong>GAT + GraphSAGE</strong> encoder with pairwise temporal
              classification, outperforming classical topological and wear-index baselines.
            </p>
          </div>

          {error && (
            <div className="card" style={{ borderColor: "#e07a5f" }}>
              <p style={{ color: "#e07a5f", fontSize: "0.85rem" }}>{error}</p>
            </div>
          )}
        </aside>
      </main>
    </div>
  );
}
