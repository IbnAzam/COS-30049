// PieSpamHam.jsx
import { useEffect, useRef } from "react";
import * as d3 from "d3";
import { useQuery } from "@tanstack/react-query";

const API = import.meta.env.VITE_API_BASE || "http://localhost:8000";
const COLORS = { Ham: "#5bc0de", Spam: "#d9534f" };

/** Viewport-safe pointer coordinates for mouse/touch */
const pointerXY = (evt) => {
  if (evt?.clientX != null) return [evt.clientX, evt.clientY]; // mouse
  const t = evt?.touches?.[0] || evt?.changedTouches?.[0];      // touch
  if (t) return [t.clientX, t.clientY];
  return [evt?.pageX ?? 0, evt?.pageY ?? 0];                    // fallback
};

/** Create/reuse a single BODY-level tooltip (true fixed positioning) */
function getBodyTooltip() {
  let tip = d3.select("#global-pie-tooltip");
  if (tip.empty()) {
    tip = d3.select("body").append("div").attr("id", "global-pie-tooltip");
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

// ---- fetcher (normalize to array of {label, c:number}) ----
async function fetchSummary() {
  const res = await fetch(`${API}/api/stats/summary`);
  if (!res.ok) throw new Error("Failed to load summary");
  const json = await res.json();
  const byLabel = Array.isArray(json?.by_label) ? json.by_label : [];
  const map = Object.fromEntries(
    byLabel.map((x) => [x.label, Number(x.c ?? 0)])
  );
  // Ensure both categories always exist and are numbers
  return [
    { label: "Ham",  c: map.Ham  ?? 0 },
    { label: "Spam", c: map.Spam ?? 0 },
  ];
}

export default function PieSpamHam({ width = 360, height = 260 }) {
  const svgRef = useRef(null);

  const { data, error, isLoading } = useQuery({
    queryKey: ["summary-pie"],
    queryFn: fetchSummary,
    refetchInterval: 10000,
    refetchOnWindowFocus: true,
    staleTime: 5000,
    placeholderData: (prev) => prev, // keep previous slice while refetching
  });

  useEffect(() => {
    // Guard: must be an array
    if (!Array.isArray(data)) return;

    const rows = data.map((d) => ({ label: d.label, c: Number(d.c) || 0 }));
    const total = rows.reduce((a, b) => a + b.c, 0);

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const tip = getBodyTooltip();

    const margin = { top: 12, right: 12, bottom: 40, left: 12 };
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;
    const radius = Math.min(w, h) / 2;

    const g = svg
      .attr("width", width)      // explicit size prevents layout flashes
      .attr("height", height)
      .attr("viewBox", `0 0 ${width} ${height}`)
      .append("g")
      .attr("transform", `translate(${margin.left + w / 2}, ${margin.top + h / 2})`);

    // If nothing to show, render an empty state (no crash)
    if (total === 0) {
      g.append("text")
        .attr("text-anchor", "middle")
        .attr("dy", "0.35em")
        .attr("fill", "#666")
        .text("No data yet");
      return;
    }

    const pie = d3.pie().sort(null).value((d) => d.c);
    const arcs = pie(rows);
    const arc = d3.arc().innerRadius(radius * 0.55).outerRadius(radius);

    const showTip = (evt, d) => {
      const [x, y] = pointerXY(evt);
      const pct = total > 0 ? ((d.data.c / total) * 100).toFixed(1) : "0.0";
      tip
        .html(
          `<div style="font-weight:600;margin-bottom:2px">${d.data.label}</div>
           <div>Count: <b>${d.data.c}</b></div>
           <div style="color:#666">Share: ${pct}%</div>`
        )
        .style("left", `${x + 12}px`)
        .style("top", `${y + 12}px`)
        .style("opacity", 1);
    };
    const moveTip = (evt) => {
      const [x, y] = pointerXY(evt);
      tip.style("left", `${x + 12}px`).style("top", `${y + 12}px`);
    };
    const hideTip = () => tip.style("opacity", 0);

    // slices
    g.selectAll("path.slice")
      .data(arcs)
      .enter()
      .append("path")
      .attr("class", "slice")
      .attr("d", arc)
      .attr("fill", (d) => COLORS[d.data.label] || "#999")
      .attr("stroke", "#fff")
      .attr("stroke-width", 1)
      .attr("tabindex", 0)
      .attr("role", "img")
      .attr("aria-label", (d) => {
        const pct = total > 0 ? ((d.data.c / total) * 100).toFixed(1) : "0.0";
        return `${d.data.label}: ${d.data.c} (${pct} percent)`;
      })
      .style("cursor", "default")
      .on("mouseenter", function (event, d) {
        d3.select(this).attr("opacity", 0.9);
        showTip(event, d);
      })
      .on("mousemove", moveTip)
      .on("mouseleave", function () {
        d3.select(this).attr("opacity", 1);
        hideTip();
      })
      .on("touchstart", function (event, d) {
        d3.select(this).attr("opacity", 0.9);
        showTip(event, d);
      })
      .on("touchmove", moveTip)
      .on("touchend", function () {
        d3.select(this).attr("opacity", 1);
        hideTip();
      })
      .on("focus", function (event, d) {
        const r = this.getBoundingClientRect();
        showTip({ clientX: r.left + r.width / 2, clientY: r.top }, d);
      })
      .on("blur", hideTip);

    // centre labels
    g.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "-0.2em")
      .attr("font-weight", 700)
      .attr("font-size", 18)
      .text(total);

    g.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "1.1em")
      .attr("fill", "#666")
      .attr("font-size", 12)
      .text("Total scans");

    // legend
    const legend = svg
      .append("g")
      .attr("transform", `translate(${(width - 180) / 2}, ${height - margin.bottom + 10})`);

    const items = rows.map((d) => ({ name: d.label, color: COLORS[d.label], c: d.c }));
    const li = legend
      .selectAll(".legend-item")
      .data(items)
      .enter()
      .append("g")
      .attr("class", "legend-item")
      .attr("transform", (_, i) => `translate(${i * 90}, 0)`);

    li.append("rect").attr("width", 12).attr("height", 12).attr("rx", 2).attr("fill", (d) => d.color);
    li.append("text").attr("x", 18).attr("y", 10).attr("font-size", 12).text((d) => d.name);
    li.append("text").attr("x", 18).attr("y", 24).attr("font-size", 11).attr("fill", "#666").text((d) => d.c);
  }, [data, width, height]);

  if (isLoading) return <p>Loading pie…</p>;
  if (error) return <p style={{ color: "crimson" }}>{error.message}</p>;

  // explicit width/height avoids layout collapses
  return <svg ref={svgRef} width={width} height={height} style={{ width: "100%", height }} />;
}
