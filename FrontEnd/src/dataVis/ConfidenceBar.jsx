// src/components/ConfidenceBar.jsx
import { useEffect, useRef } from "react";
import * as d3 from "d3";

export default function ConfidenceBar({ probability }) {
  const ref = useRef(null);
  // accept 0..1 or 0..100
  const p = probability > 1 ? probability / 100 : probability;

  useEffect(() => {
    const width = 320, height = 26, radius = 6;
    const svg = d3.select(ref.current)
      .attr("viewBox", `0 0 ${width} ${height}`)
      .attr("role", "img")
      .attr("aria-label", `Confidence ${(p*100).toFixed(1)} percent`);

    svg.selectAll("*").remove();

    // background
    svg.append("rect")
      .attr("x", 0).attr("y", 0)
      .attr("width", width).attr("height", height)
      .attr("rx", radius).attr("ry", radius)
      .attr("fill", "#eee");

    // foreground (animated width)
    const fg = svg.append("rect")
      .attr("x", 0).attr("y", 0)
      .attr("height", height)
      .attr("rx", radius).attr("ry", radius)
      .attr("fill", "#4a7bd8")
      .attr("width", 0);

    fg.transition().duration(500).attr("width", width * p);

    // text
    svg.append("text")
      .attr("x", width - 8)
      .attr("y", height / 2 + 4)
      .attr("text-anchor", "end")
      .attr("font-size", 12)
      .attr("fill", "#222")
      .text(`${(p * 100).toFixed(1)}%`);
  }, [p]);

  return <svg ref={ref} style={{ width: "100%", maxWidth: 360 }} />;
}
