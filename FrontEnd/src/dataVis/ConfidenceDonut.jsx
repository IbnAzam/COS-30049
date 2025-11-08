// src/components/ConfidenceDonut.jsx
import { useEffect, useRef } from "react";
import * as d3 from "d3";

export default function ConfidenceDonut({ probability, label="Spam" }) {
  const ref = useRef(null);
  const p = probability > 1 ? probability / 100 : probability;

  useEffect(() => {
    const size = 160, thickness = 18;
    const radius = size / 2;
    const svg = d3.select(ref.current)
      .attr("viewBox", `0 0 ${size} ${size}`)
      .attr("role", "img")
      .attr("aria-label", `${label} probability ${(p*100).toFixed(1)} percent`);

    svg.selectAll("*").remove();

    const g = svg.append("g").attr("transform", `translate(${radius},${radius})`);

    const arc = d3.arc().innerRadius(radius - thickness).outerRadius(radius);
    const bg = { startAngle: 0, endAngle: 2 * Math.PI };
    g.append("path").attr("d", arc(bg)).attr("fill", "#eee");

    const fg = g.append("path").attr("fill", "#4a7bd8");
    fg.transition()
      .duration(700)
      .attrTween("d", () => {
        const i = d3.interpolate(0, 2 * Math.PI * p);
        return t => arc({ startAngle: 0, endAngle: i(t) });
      });

    // center text
    g.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "-4")
      .attr("font-size", 18)
      .attr("font-weight", 700)
      .text(`${(p * 100).toFixed(1)}%`);

    g.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "16")
      .attr("font-size", 12)
      .attr("fill", "#555")
      .text(`${label} confidence`);
  }, [p, label]);

  return <svg ref={ref} style={{ width: 180, height: 180 }} />;
}
