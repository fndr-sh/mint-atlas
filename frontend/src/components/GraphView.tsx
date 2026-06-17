import { useEffect, useRef } from "react";
import * as d3 from "d3";

export interface GraphNode {
  id: string;
  side: string;
  wear_index: number;
  estimated_sequence: number | null;
}

export interface GraphEdge {
  source: string;
  target: string;
  coin_count?: number;
}

export interface WearTransfer {
  source: string;
  target: string;
  strength: number;
}

interface Props {
  nodes: GraphNode[];
  edges: GraphEdge[];
  wearTransfers: WearTransfer[];
  chronology?: string[];
}

interface SimNode extends d3.SimulationNodeDatum {
  id: string;
  side: string;
  wear_index: number;
  seq: number;
}

interface SimLink extends d3.SimulationLinkDatum<SimNode> {
  type: "coin" | "wear";
}

export default function GraphView({ nodes, edges, wearTransfers, chronology }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || nodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const width = svgRef.current.clientWidth;
    const height = svgRef.current.clientHeight;

    const seqMap = new Map(chronology?.map((id, i) => [id, i]) ?? []);
    const simNodes: SimNode[] = nodes.map((n) => ({
      id: n.id,
      side: n.side,
      wear_index: n.wear_index,
      seq: seqMap.get(n.id) ?? n.estimated_sequence ?? 0,
    }));

    const nodeMap = new Map(simNodes.map((n) => [n.id, n]));

    const simLinks: SimLink[] = [
      ...edges.map((e) => ({
        source: nodeMap.get(e.source)!,
        target: nodeMap.get(e.target)!,
        type: "coin" as const,
      })),
      ...wearTransfers.map((w) => ({
        source: nodeMap.get(w.source)!,
        target: nodeMap.get(w.target)!,
        type: "wear" as const,
      })),
    ].filter((l) => l.source && l.target);

    const g = svg.append("g");

    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.2, 4])
      .on("zoom", (event) => g.attr("transform", event.transform));
    svg.call(zoom);

    const simulation = d3
      .forceSimulation(simNodes)
      .force(
        "link",
        d3
          .forceLink(simLinks)
          .id((d) => (d as SimNode).id)
          .distance((l) => ((l as SimLink).type === "wear" ? 80 : 120))
      )
      .force("charge", d3.forceManyBody().strength(-280))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide(22));

    const defs = svg.append("defs");
    defs
      .append("marker")
      .attr("id", "arrow")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 22)
      .attr("refY", 0)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-5L10,0L0,5")
      .attr("fill", "var(--accent)");

    const link = g
      .append("g")
      .selectAll("line")
      .data(simLinks)
      .join("line")
      .attr("stroke", (d) => (d.type === "wear" ? "var(--accent)" : "var(--border)"))
      .attr("stroke-width", (d) => (d.type === "wear" ? 2 : 1))
      .attr("stroke-opacity", (d) => (d.type === "wear" ? 0.8 : 0.4))
      .attr("marker-end", (d) => (d.type === "wear" ? "url(#arrow)" : null));

    const node = g
      .append("g")
      .selectAll<SVGGElement, SimNode>("g")
      .data(simNodes)
      .join("g")
      .call(
        d3.drag<SVGGElement, SimNode>().on("start", (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        }).on("drag", (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        }).on("end", (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        }) as unknown as (selection: d3.Selection<SVGGElement, SimNode, SVGGElement, unknown>) => void
      );

    node
      .append("circle")
      .attr("r", 14)
      .attr("fill", (d) => (d.side === "obverse" ? "var(--obverse)" : "var(--reverse)"))
      .attr("fill-opacity", 0.85)
      .attr("stroke", "var(--bg)")
      .attr("stroke-width", 2);

    node
      .append("text")
      .text((d) => d.id)
      .attr("text-anchor", "middle")
      .attr("dy", 28)
      .attr("fill", "var(--muted)")
      .attr("font-size", "9px")
      .attr("font-family", "JetBrains Mono, monospace");

    simulation.on("tick", () => {
      link
        .attr("x1", (d) => (d.source as SimNode).x!)
        .attr("y1", (d) => (d.source as SimNode).y!)
        .attr("x2", (d) => (d.target as SimNode).x!)
        .attr("y2", (d) => (d.target as SimNode).y!);
      node.attr("transform", (d) => `translate(${d.x},${d.y})`);
    });

    return () => {
      simulation.stop();
    };
  }, [nodes, edges, wearTransfers, chronology]);

  return <svg ref={svgRef} />;
}
