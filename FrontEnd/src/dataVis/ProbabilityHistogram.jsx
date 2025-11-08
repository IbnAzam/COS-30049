// ProbabilityHistogram.jsx
import { useEffect, useRef } from "react";
import * as d3 from "d3";
import { useQuery } from "@tanstack/react-query";

const API = import.meta.env.VITE_API_BASE || "http://localhost:8000";

// ---- zoom-safe tooltip under <body> ----
const pointerXY = (evt) => {
  if (evt?.clientX != null) return [evt.clientX, evt.clientY];
  const t = evt?.touches?.[0] || evt?.changedTouches?.[0];
  if (t) return [t.clientX, t.clientY];
  return [evt?.pageX ?? 0, evt?.pageY ?? 0];
};

function getBodyTooltip() {
  let tip = d3.select("#global-prob-hist-tooltip");
  if (tip.empty()) {
    tip = d3.select("body").append("div").attr("id", "global-prob-hist-tooltip");
  }
  return tip
    .style("position", "fixed")
    .style("pointer-events", "none")
    .style("background", "rgba(255,255,255,0.95)")
    .style("box-shadow", "0 1px 2px rgba(0,0,0,0.06), 0 4px 12px rgba(0,0,0,0.08)")
    .style("border", "1px solid #e5e7eb")
    .style("border-radius", "8px")
    .style("padding", "8px 10px")
    .style("font-size", "12px")
    .style("line-height", "1.2")
    .style("z-index", 9999)
    .style("opacity", 0);
}

// Single-hue teal → red gradient across probability 0→1
const colorForProb = d3
  .scaleLinear()
  .domain([0, 1])
  .range(["#1dc7ff", "#ff2222"])
  .interpolate(d3.interpolateRgb);

// Format a [lo, hi) bin range string like "0.4–0.5"
const fmtRange = (lo, hi) => `${d3.format(".1f")(lo)}–${d3.format(".1f")(hi)}`;

// ---- Fetcher ----
async function fetchDistribution(bins) {
  const res = await fetch(`${API}/api/stats/distribution?bins=${bins}`);
  if (!res.ok) throw new Error("Failed to load distribution");
  const json = await res.json();

  let counts = json?.counts ?? [];
  let n = counts.length;

  // ✅ Fallback edges when server doesn't supply them
  let edges =
    json?.bin_edges && json.bin_edges.length === n + 1
      ? json.bin_edges
      : (n > 0 ? Array.from({ length: n + 1 }, (_, i) => i / n) : [0, 1]);

  // ✅ If there are no counts at all, synthesize a harmless zero bin
  if (n === 0) {
    counts = [0];
    edges = [0, 1];
    n = 1;
  }

  const total = json?.total ?? counts.reduce((a, b) => a + b, 0);

  return { counts, edges, total };
}


export default function ProbabilityHistogram({ width = 720, height = 320, bins = 20 }) {
  const svgRef = useRef(null);

  // ✅ Replace manual fetch with React Query
  const { data, error, isLoading } = useQuery({
    queryKey: ["distribution", bins],
    queryFn: () => fetchDistribution(bins),
    refetchInterval: 10000, // refresh every 10s
    refetchOnWindowFocus: true,
    staleTime: 5000,
  });

  useEffect(() => {
    if (!data) return;

    const { counts, edges, total } = data;
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const margin = { top: 12, right: 80, bottom: 40, left: 60 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const g = svg
      .attr("viewBox", `0 0 ${width} ${height}`)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Build bin centers from edges
    const centers = data.counts.map((_, i) => (data.edges[i] + data.edges[i + 1]) / 2);

    // Scales
    const x = d3.scaleBand().domain(centers).range([0, innerW]).padding(0.1);
    const yMax = d3.max(counts) ?? 1;
    const y = d3.scaleLinear().domain([0, yMax]).nice().range([innerH, 0]);

    // Decision boundary line (x=0.5)
    g.append("line")
      .attr("x1", innerW * 0.5)
      .attr("x2", innerW * 0.5)
      .attr("y1", 0)
      .attr("y2", innerH)
      .attr("stroke", "#999")
      .attr("stroke-dasharray", "4 4");

    g.append("text")
      .attr("x", innerW * 0.5 - 60)
      .attr("y", -10)
      .attr("fill", "#666")
      .attr("font-size", 11)
      .text("Decision boundary (0.5)");

    // Tooltip
    const tip = getBodyTooltip();
    const showTip = (evt, i) => {
      const lo = edges[i];
      const hi = edges[i + 1];
      const c = counts[i];
      const pct = total > 0 ? ((c / total) * 100).toFixed(1) : "0.0";
      tip
        .html(
          `<div style="font-weight:600;margin-bottom:2px">Probability ${fmtRange(lo, hi)}</div>
           <div>Count: <b>${c}</b></div>
           <div style="color:#666">Share: ${pct}%</div>`
        )
        .style("opacity", 1);
      const [xv, yv] = pointerXY(evt);
      tip.style("left", `${xv + 12}px`).style("top", `${yv + 12}px`);
    };
    const moveTip = (evt) => {
      const [xv, yv] = pointerXY(evt);
      tip.style("left", `${xv + 12}px`).style("top", `${yv + 12}px`);
    };
    const hideTip = () => tip.style("opacity", 0);

    // Bars
    // Bars (bind both count and index)
    g.selectAll("rect.bar")
      .data(counts.map((c, i) => ({ c, i })))
      .enter()
      .append("rect")
      .attr("class", "bar")
      .attr("x", (d) => x(centers[d.i]))
      .attr("width", x.bandwidth())
      .attr("y", innerH)
      .attr("height", 0)
      .attr("fill", (d) => colorForProb(centers[d.i]))
      .style("cursor", "default")
      .on("mouseenter", function (event, d) {
        d3.select(this).attr("opacity", 0.9);
        showTip(event, d.i);     // ✅ pass the real index
      })
      .on("mousemove", moveTip)
      .on("mouseleave", function () {
        d3.select(this).attr("opacity", 1);
        hideTip();
      })
      .transition()
      .duration(650)
      .attr("y", (d) => y(d.c))
      .attr("height", (d) => innerH - y(d.c));


    // Axis
    const xAxis = d3
      .axisBottom(x)
      .tickValues(
        centers.filter((_, i) => (edges.length - 1) <= 12 ? true : i % 2 === 0)
      )
      .tickFormat((c) => d3.format(".1f")(c));
    const yAxis = d3.axisLeft(y).ticks(5).tickFormat(d3.format("~s"));

    g.append("g")
      .attr("transform", `translate(0,${innerH})`)
      .call(xAxis)
      .selectAll("text")
      .attr("font-size", 12);

    g.append("g").call(yAxis).selectAll("text").attr("font-size", 12);

    // Axis labels
    g.append("text")
      .attr("x", innerW / 2)
      .attr("y", innerH + 32)
      .attr("text-anchor", "middle")
      .attr("font-size", 12)
      .attr("fill", "#444")
      .text("Prediction probability");

    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -innerH / 2)
      .attr("y", -36)
      .attr("text-anchor", "middle")
      .attr("font-size", 12)
      .attr("fill", "#444")
      .text("Count of predictions");
  }, [data, width, height]);

  if (isLoading) return <p>Loading histogram…</p>;
  if (error) return <p style={{ color: "crimson" }}>{error.message}</p>;

  return <svg ref={svgRef} width={width} height={height} style={{ width: "100%", height }} />;

}
