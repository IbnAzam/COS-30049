// StackedBar.jsx (7/14/30 toggle, no jitter; debounced "Refreshing…" badge)
import { useEffect, useMemo, useRef, useState } from "react";
import * as d3 from "d3";
import { useQuery } from "@tanstack/react-query";

const API = import.meta.env.VITE_API_BASE || "http://localhost:8000";

/* ---------- helpers ---------- */
const isLocalDayKey = (key) =>
  typeof key === "string" && key.length === 10 && !key.includes("T");

const melDayKey = (d) =>
  d.toLocaleDateString("en-CA", { timeZone: "Australia/Melbourne" });

const buildLastNMelKeys = (n) => {
  const todayKey = melDayKey(new Date());
  const base = new Date(`${todayKey}T00:00:00`);
  const out = [];
  for (let i = n - 1; i >= 0; i--) {
    const d = new Date(base);
    d.setDate(base.getDate() - i);
    out.push(melDayKey(d));
  }
  return out;
};

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

const pointerXY = (evt) => {
  if (evt?.clientX != null) return [evt.clientX, evt.clientY];
  const t = evt?.touches?.[0] || evt?.changedTouches?.[0];
  if (t) return [t.clientX, t.clientY];
  return [evt?.pageX ?? 0, evt?.pageY ?? 0];
};

function getBodyTooltip() {
  let tip = d3.select("#global-stackedbar-tooltip");
  if (tip.empty()) tip = d3.select("body").append("div").attr("id", "global-stackedbar-tooltip");
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

/* ---------- fetcher ---------- */
async function fetchTimeseries(days) {
  const res = await fetch(
    `${API}/api/stats/timeseries?bucket=day&days=${days}&tz=Australia/Melbourne`
  );
  if (!res.ok) throw new Error("Failed to load timeseries");
  const json = await res.json();
  return json?.points ?? [];
}

/* ---------- component ---------- */
export default function StackedBar({ width = 720, height = 320 }) {
  const [days, setDays] = useState(7);
  const ref = useRef(null);

  const { data, error, isLoading, isFetching } = useQuery({
    queryKey: ["timeseries", days],
    queryFn: () => fetchTimeseries(days),
    refetchInterval: 10000, // 10s
    refetchOnWindowFocus: true,
    staleTime: 5000,
    keepPreviousData: true,
  });

  // Normalize N-day points
  const points = useMemo(() => {
    if (!data) return null;

    if (data.length === 0) {
      return buildLastNMelKeys(days).map((k) => ({
        bucket: k,
        Spam: 0,
        Ham: 0,
        Total: 0,
      }));
    }

    const serverKeySample = data[0].bucket;
    const serverUsesLocal = isLocalDayKey(serverKeySample);

    if (serverUsesLocal) {
      const map = new Map(data.map((d) => [d.bucket, d]));
      return buildLastNMelKeys(days).map((k) => {
        const hit = map.get(k);
        const Spam = hit?.Spam ?? 0;
        const Ham = hit?.Ham ?? 0;
        return { bucket: k, Spam, Ham, Total: Spam + Ham };
      });
    }

    // Fallback: UTC ISO keys
    const map = new Map(data.map((d) => [d.bucket, d]));
    const out = [];
    const todayUTC = new Date();
    todayUTC.setUTCHours(0, 0, 0, 0);

    for (let i = days - 1; i >= 0; i--) {
      const d = new Date(todayUTC);
      d.setUTCDate(todayUTC.getUTCDate() - i);
      const iso = d.toISOString().replace(".000Z", "Z");
      const hit = map.get(iso);
      const Spam = hit?.Spam ?? 0;
      const Ham = hit?.Ham ?? 0;
      out.push({ bucket: iso, Spam, Ham, Total: Spam + Ham });
    }
    return out;
  }, [data, days]);

  // Signatures for change detection
  const valueSig = useMemo(() => {
    if (!points) return "";
    return points.map((p) => `${p.bucket}:${p.Ham},${p.Spam}`).join("|");
  }, [points]);

  const bucketsSig = useMemo(() => {
    if (!points) return "";
    return points.map((p) => p.bucket).join("|");
  }, [points]);

  const lastValueSigRef = useRef("");
  const lastBucketsSigRef = useRef("");

  // Debounced "Refreshing…" badge (no blink)
  const [showBadge, setShowBadge] = useState(false);
  const firstLoadRef = useRef(true);
  useEffect(() => {
    // ignore initial load (handled by "Loading chart…")
    if (firstLoadRef.current && isLoading) return;
    if (firstLoadRef.current && !isLoading) firstLoadRef.current = false;

    let showTimer, hideTimer;
    if (isFetching) {
      showTimer = setTimeout(() => setShowBadge(true), 500); // show only if fetch > 500ms
    } else {
      hideTimer = setTimeout(() => setShowBadge(false), 250); // linger slightly to avoid flicker
    }
    return () => {
      clearTimeout(showTimer);
      clearTimeout(hideTimer);
    };
  }, [isFetching, isLoading]);

  // Build/update chart only when buckets/values changed (prevents jitter)
  useEffect(() => {
    if (!points) return;

    const prevVal = lastValueSigRef.current;
    const prevBuck = lastBucketsSigRef.current;

    const bucketsChanged = prevBuck && prevBuck !== bucketsSig;
    const valuesChanged = prevVal && prevVal !== valueSig;

    // If nothing changed at all, do nothing (no DOM touches)
    if (prevVal && prevVal === valueSig && prevBuck === bucketsSig) {
      return;
    }

    lastValueSigRef.current = valueSig;
    lastBucketsSigRef.current = bucketsSig;

    const shouldAnimate = !prevVal || valuesChanged || bucketsChanged;

    const svg = d3.select(ref.current);
    svg.selectAll("*").remove(); // full rebuild only on meaningful changes

    const tooltip = getBodyTooltip();

    const margin = { top: 16, right: 12, bottom: 44, left: 48 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    const g = svg
      .attr("viewBox", `0 0 ${width} ${height}`)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const x = d3
      .scaleBand()
      .domain(points.map((d) => d.bucket))
      .range([0, innerW])
      .padding(0.2);

    const yMax = d3.max(points, (d) => d.Total) ?? 0;
    const y = d3.scaleLinear().domain([0, yMax === 0 ? 1 : yMax]).nice().range([innerH, 0]);

    const keys = ["Ham", "Spam"];
    const color = d3.scaleOrdinal().domain(keys).range(["#5bc0de", "#d9534f"]);
    const stacked = d3.stack().keys(keys)(points);

    // Tooltip handlers
    const showTip = (event, d, key) => {
      const bucketKey = d.data.bucket;
      const count = Math.round(d[1] - d[0]);
      const total = d.data.Total || 0;
      const share = total ? ((count / total) * 100).toFixed(1) : "0.0";
      const html = `
        <div style="font-weight:600;margin-bottom:2px">${fmtTick(bucketKey)}</div>
        <div>
          <span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${color(
            key
          )};margin-right:6px;vertical-align:middle;"></span>${key}: <b>${count}</b>
        </div>
        <div style="color:#666">Share: ${share}%</div>
        <div style="margin-top:2px;color:#666">Total: ${total}</div>
      `;
      const [xv, yv] = pointerXY(event);
      tooltip.html(html).style("left", `${xv + 12}px`).style("top", `${yv + 12}px`).style("opacity", 1);
    };
    const moveTip = (event) => {
      const [xv, yv] = pointerXY(event);
      tooltip.style("left", `${xv + 12}px`).style("top", `${yv + 12}px`);
    };
    const hideTip = () => tooltip.style("opacity", 0);

    // Bars
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
      .duration(shouldAnimate ? 600 : 0)
      .attr("y", (d) => y(d[1]))
      .attr("height", (d) => y(d[0]) - y(d[1]));

    // Axes (thin ticks when many days)
    const tickValues =
      points.length <= 10
        ? points.map((d) => d.bucket)
        : points.map((d) => d.bucket).filter((_, i) => i % Math.ceil(points.length / 10) === 0);

    const xAxis = d3.axisBottom(x).tickValues(tickValues).tickFormat((b) => fmtTick(b));
    const yAxis = d3.axisLeft(y).ticks(5).tickFormat(d3.format("~s"));

    g.append("g")
      .attr("transform", `translate(0,${innerH})`)
      .call(xAxis)
      .selectAll("text")
      .attr("font-size", 12);

    g.append("g").call(yAxis).selectAll("text").attr("font-size", 12);

    // Y label
    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -innerH / 2)
      .attr("y", -margin.left + 14)
      .attr("text-anchor", "middle")
      .attr("font-size", 12)
      .attr("fill", "#444")
      .text("Count");

    // Legend
    const legendHeight = 20;
    const legend = g
      .append("g")
      .attr(
        "transform",
        `translate(${innerW / 2 - (keys.length * 80) / 2}, ${innerH + margin.bottom - legendHeight})`
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
  }, [points, width, height, valueSig, bucketsSig]);

  if (isLoading && !points) return <p>Loading chart…</p>;
  if (error) return <p style={{ color: "crimson" }}>{error.message}</p>;

  return (
    <div style={{ position: "relative" }}>
      {/* Toggle */}
      <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
        {[7, 14, 30].map((n) => (
          <button
            key={n}
            onClick={() => setDays(n)}
            style={{
              padding: "6px 10px",
              borderRadius: 8,
              border: n === days ? "2px solid #1976d2" : "1px solid #ccc",
              background: n === days ? "#e8f1fd" : "#fff",
              cursor: "pointer",
              fontSize: 12,
            }}
          >
            {n} days
          </button>
        ))}
      </div>

      {/* Debounced, absolute overlay (no layout shift, no flicker) */}
      {showBadge && (
        <div
          style={{
            position: "absolute",
            right: 4,
            top: 0,
            fontSize: 11,
            color: "#888",
            pointerEvents: "none",
          }}
        >
          Refreshing…
        </div>
      )}

      <svg ref={ref} width={width} height={height} style={{ width: "100%", height }} />
    </div>
  );
}
