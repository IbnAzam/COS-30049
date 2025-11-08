// StackedBar7d.jsx
import { useEffect, useMemo, useRef, useState } from "react";
import * as d3 from "d3";

const API = import.meta.env.VITE_API_BASE || "http://localhost:8000";

// ---- Time formatting helpers ----

// Melbourne label helper (display only)
const fmtMelDate = (iso) =>
  new Date(iso).toLocaleDateString("en-AU", {
    timeZone: "Australia/Melbourne",
    month: "short",
    day: "numeric",
    weekday: "short",
  });

// Detect if a bucket key is a local day key like "2025-11-09"
const isLocalDayKey = (key) =>
  typeof key === "string" && key.length === 10 && !key.includes("T");

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

const fmtTick = (bucket) =>
  isLocalDayKey(bucket) ? labelFromLocalKey(bucket) : labelFromUtcIso(bucket);

// ---- Tooltip utils (zoom-safe) ----

/** Viewport-safe pointer coordinates for mouse/touch */
const pointerXY = (evt) => {
  if (evt?.clientX != null) return [evt.clientX, evt.clientY]; // mouse
  const t = evt?.touches?.[0] || evt?.changedTouches?.[0]; // touch
  if (t) return [t.clientX, t.clientY];
  return [evt?.pageX ?? 0, evt?.pageY ?? 0]; // fallback
};

/** Create/reuse a single BODY-level tooltip (true fixed positioning) */
function getBodyTooltip() {
  let tip = d3.select("#global-stackedbar-tooltip");
  if (tip.empty()) {
    tip = d3.select("body").append("div").attr("id", "global-stackedbar-tooltip");
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

// -----------------------------------

export default function StackedBar7d({ width = 720, height = 320 }) {
  const [data, setData] = useState(null);
  const [err, setErr] = useState("");
  const ref = useRef(null);

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(
          `${API}/api/stats/timeseries?bucket=day&days=7&tz=Australia/Melbourne`
        );
        const json = await res.json();
        setData(json?.points ?? []);
      } catch (e) {
        setErr(e.message || "Failed to load timeseries");
      }
    })();
  }, []);

  // Ensure 7 consecutive buckets that match what the server sends.
  // If server sends local day keys ("YYYY-MM-DD"): build last 7 Melbourne days and map counts.
  // If server sends UTC ISO (fallback): keep the original UTC path.
  const points7 = useMemo(() => {
    if (!data) return null;

    if (data.length === 0) {
      // keep axes stable with 7 zero bars (local-day version)
      return buildLast7MelKeys().map((k) => ({
        bucket: k,
        Spam: 0,
        Ham: 0,
        Total: 0,
      }));
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

    const tooltip = getBodyTooltip();

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
    const color = d3.scaleOrdinal().domain(keys).range(["#5bc0de", "#d9534f"]); // Ham teal, Spam red

    const stacked = d3.stack().keys(keys)(points7);

    // Tooltip handlers (zoom-safe)
    const showTip = (event, d, key) => {
      const bucketKey = d.data.bucket;
      const count = Math.round(d[1] - d[0]); // segment size
      const html = `
        <div style="font-weight:600;margin-bottom:2px">${fmtTick(bucketKey)}</div>
        <div>
          <span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${color(
            key
          )};margin-right:6px;vertical-align:middle;"></span>${key}: <b>${count}</b>
        </div>
        <div style="margin-top:2px;color:#666">Total: ${d.data.Total}</div>
      `;
      const [xv, yv] = pointerXY(event);
      tooltip.html(html).style("left", `${xv + 12}px`).style("top", `${yv + 12}px`).style("opacity", 1);
    };

    const moveTip = (event) => {
      const [xv, yv] = pointerXY(event);
      tooltip.style("left", `${xv + 12}px`).style("top", `${yv + 12}px`);
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

    g.append("g").call(yAxis).selectAll("text").attr("font-size", 12);

    // Legend (bottom-center)
    const legendHeight = 20;
    const legend = g
      .append("g")
      .attr(
        "transform",
        `translate(${innerW / 2 - (keys.length * 80) / 2}, ${
          innerH + margin.bottom - legendHeight
        })`
      );

    const items = keys.map((name) => ({ name, color: color(name) }));
    const item = legend
      .selectAll(".legend-item")
      .data(items)
      .enter()
      .append("g")
      .attr("class", "legend-item")
      .attr("transform", (_, i) => `translate(${i * 80}, 0)`);

    item.append("rect").attr("width", 12).attr("height", 12).attr("rx", 2).attr("fill", (d) => d.color);

    item.append("text").attr("x", 18).attr("y", 10).attr("font-size", 12).text((d) => d.name);
  }, [points7, width, height]);

  if (err) return <p style={{ color: "crimson" }}>{err}</p>;
  if (!points7) return <p>Loading chart…</p>;

  return <svg ref={ref} style={{ width: "100%", height }} />;
}
