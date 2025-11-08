import { useEffect, useMemo, useRef, useState } from "react";
import * as d3 from "d3";

const API = import.meta.env.VITE_API_BASE || "http://localhost:8000";

// Melbourne label helper (display only)
const fmtMelDate = (iso) =>
  new Date(iso).toLocaleDateString("en-AU", {
    timeZone: "Australia/Melbourne",
    month: "short",
    day: "numeric",
    weekday: "short",
  });


// Detect if a bucket key is a local day key like "2025-11-09"
const isLocalDayKey = (key) => typeof key === "string" && key.length === 10 && !key.includes("T");

// Melbourne "YYYY-MM-DD" from a Date
const melDayKey = (d) =>
  d.toLocaleDateString("en-CA", { timeZone: "Australia/Melbourne" });

// Build the last 7 Melbourne day keys (including today in Melbourne)
const buildLast7MelKeys = () => {
  const todayKey = melDayKey(new Date());
  const base = new Date(`${todayKey}T00:00:00`); // only used for +/- days
  const out = [];
  for (let i = 6; i >= 0; i--) {
    const d = new Date(base);
    d.setDate(base.getDate() - i);
    out.push(melDayKey(d)); // always recompute via Melbourne tz
  }
  return out;
};

// Labels for axis/tooltip that work for both formats
const labelFromLocalKey = (key) =>
  new Date(key + "T00:00:00").toLocaleDateString("en-AU", {
    timeZone: "Australia/Melbourne",
    month: "short",
    day: "numeric",
    weekday: "short",
  });

const labelFromUtcIso = (iso) =>
  new Date(iso).toLocaleDateString("en-AU", {
    timeZone: "Australia/Melbourne",
    month: "short",
    day: "numeric",
    weekday: "short",
  });

const fmtTick = (bucket) => (isLocalDayKey(bucket) ? labelFromLocalKey(bucket) : labelFromUtcIso(bucket));




export default function StackedBar7d({ width = 720, height = 320 }) {
  const [data, setData] = useState(null);
  const [err, setErr] = useState("");
  const ref = useRef(null);
  const tooltipRef = useRef(null);

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API}/api/stats/timeseries?bucket=day&days=7&tz=Australia/Melbourne`);
        const json = await res.json();
        setData(json?.points ?? []);
      } catch (e) {
        setErr(e.message || "Failed to load timeseries");
      }
    })();
  }, []);

  // Ensure 7 consecutive UTC-midnight buckets, fill gaps with zeros
  // Ensure 7 consecutive buckets that match what the server sends.
// If server sends local day keys ("YYYY-MM-DD"): build last 7 Melbourne days and map counts.
// If server sends UTC ISO (fallback): keep your original UTC path.
const points7 = useMemo(() => {
  if (!data) return null;

  if (data.length === 0) {
    // keep axes stable with 7 zero bars (local-day version)
    return buildLast7MelKeys().map((k) => ({ bucket: k, Spam: 0, Ham: 0, Total: 0 }));
  }

  const serverKeySample = data[0].bucket;
  const serverUsesLocal = isLocalDayKey(serverKeySample);

  if (serverUsesLocal) {
    // Server has already aggregated by Melbourne day ("YYYY-MM-DD")
    const map = new Map(data.map((d) => [d.bucket, d]));
    return buildLast7MelKeys().map((k) => {
      const hit = map.get(k);
      const Spam = hit?.Spam ?? 0;
      const Ham = hit?.Ham ?? 0;
      return { bucket: k, Spam, Ham, Total: Spam + Ham };
    });
  }

  // Fallback: server is still sending UTC ISO midnights
  const map = new Map(data.map((d) => [d.bucket, d]));
  const out = [];
  const todayUTC = new Date();
  todayUTC.setUTCHours(0, 0, 0, 0);

  for (let i = 6; i >= 0; i--) {
    const d = new Date(todayUTC);
    d.setUTCDate(todayUTC.getUTCDate() - i);
    const iso = d.toISOString().replace(".000Z", "Z");
    const hit = map.get(iso);
    const Spam = hit?.Spam ?? 0;
    const Ham = hit?.Ham ?? 0;
    out.push({ bucket: iso, Spam, Ham, Total: Spam + Ham });
  }
  return out;
}, [data]);


  useEffect(() => {
    if (!points7) return;

    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();

    // Create / reuse absolutely-positioned tooltip div (outside SVG)
    let tooltip = d3.select(tooltipRef.current);
    if (tooltip.empty()) {
      // Shouldn't happen since we render the div in JSX, but keep it safe
      tooltip = d3
        .select("body")
        .append("div")
        .attr("class", "stackedbar-tooltip");
    }

    const margin = { top: 16, right: 12, bottom: 40, left: 42 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const g = svg
      .attr("viewBox", `0 0 ${width} ${height}`)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Scales
    const x = d3
      .scaleBand()
      .domain(points7.map((d) => d.bucket))
      .range([0, innerW])
      .padding(0.2);

    const yMax = d3.max(points7, (d) => d.Total) ?? 0;
    const y = d3
      .scaleLinear()
      .domain([0, yMax === 0 ? 1 : yMax])
      .nice()
      .range([innerH, 0]);

    const keys = ["Ham", "Spam"]; // order matters for stack & legend
    const color = d3
      .scaleOrdinal()
      .domain(keys)
      .range(["#5bc0de", "#d9534f"]); // Ham teal, Spam red

    const stacked = d3.stack().keys(keys)(points7);

    // Tooltip handlers
    const showTip = (event, d, key) => {
      const bucketISO = d.data.bucket;
      const count = Math.round(d[1] - d[0]); // segment size
      const html = `
        <div style="font-weight:600;margin-bottom:2px">${fmtTick(bucketISO)}</div>
        <div><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${color(
          key
        )};margin-right:6px;vertical-align:middle;"></span>${key}: <b>${count}</b></div>
        <div style="margin-top:2px;color:#666">Total: ${d.data.Total}</div>
      `;
      const [pageX, pageY] = [event.pageX, event.pageY];
      tooltip
        .html(html)
        .style("left", `${pageX + 12}px`)
        .style("top", `${pageY + 12}px`)
        .style("opacity", 1);
    };

    const moveTip = (event) => {
      const [pageX, pageY] = [event.pageX, event.pageY];
      tooltip.style("left", `${pageX + 12}px`).style("top", `${pageY + 12}px`);
    };

    const hideTip = () => {
      tooltip.style("opacity", 0);
    };

    // Layers + rects
    const groups = g
      .selectAll("g.layer")
      .data(stacked, (d) => d.key)
      .enter()
      .append("g")
      .attr("class", (d) => `layer ${d.key}`)
      .attr("fill", (d) => color(d.key));

    groups
      .selectAll("rect")
      .data((d) => d, (d) => d.data.bucket)
      .enter()
      .append("rect")
      .attr("x", (d) => x(d.data.bucket))
      .attr("width", x.bandwidth())
      .attr("y", innerH)
      .attr("height", 0)
      .style("cursor", "default")
      .on("mouseenter", function (event, d) {
        const key = d3.select(this.parentNode).datum().key;
        d3.select(this).attr("opacity", 0.85);
        showTip(event, d, key);
      })
      .on("mousemove", moveTip)
      .on("mouseleave", function () {
        d3.select(this).attr("opacity", 1);
        hideTip();
      })
      .transition()
      .duration(600)
      .attr("y", (d) => y(d[1]))
      .attr("height", (d) => y(d[0]) - y(d[1]));

    // Axes
    const xAxis = d3.axisBottom(x).tickFormat((bucket) => fmtTick(bucket));
    const yAxis = d3.axisLeft(y).ticks(5).tickFormat(d3.format("~s"));

    g.append("g")
      .attr("transform", `translate(0,${innerH})`)
      .call(xAxis)
      .selectAll("text")
      .attr("font-size", 12);

    g.append("g")
      .call(yAxis)
      .selectAll("text")
      .attr("font-size", 12);

    // Legend
    // Place the legend at the bottom-center of the chart
    const legendHeight = 20;
    const legend = g
    .append("g")
    .attr(
        "transform",
        `translate(${innerW / 2 - (keys.length * 80) / 2}, ${innerH + margin.bottom - legendHeight})`
    );

    // Data for each key/color pair
    const items = keys.map((name) => ({ name, color: color(name) }));

    // Create groups for each legend item
    const item = legend
    .selectAll(".legend-item")
    .data(items)
    .enter()
    .append("g")
    .attr("class", "legend-item")
    .attr("transform", (_, i) => `translate(${i * 80}, 0)`); // horizontal spacing

    // Colored boxes
    item
    .append("rect")
    .attr("width", 12)
    .attr("height", 12)
    .attr("rx", 2)
    .attr("fill", (d) => d.color);

    // Labels
    item
    .append("text")
    .attr("x", 18)
    .attr("y", 10)
    .attr("font-size", 12)
    .text((d) => d.name);

  }, [points7, width, height]);

  if (err) return <p style={{ color: "crimson" }}>{err}</p>;
  if (!points7) return <p>Loading chart…</p>;

  return (
    <>
      {/* Tooltip div lives outside SVG for easy positioning */}
      <div
        ref={tooltipRef}
        style={{
          position: "fixed",
          pointerEvents: "none",
          background: "rgba(255,255,255,0.95)",
          boxShadow:
            "0 1px 2px rgba(0,0,0,0.06), 0 4px 12px rgba(0,0,0,0.08)",
          border: "1px solid #e5e7eb",
          borderRadius: 8,
          padding: "8px 10px",
          fontSize: 12,
          lineHeight: 1.2,
          opacity: 0,
          zIndex: 50,
        }}
      />
      <svg ref={ref} style={{ width: "100%", height }} />
    </>
  );
}
