/* Charts drawn with d3 (vendored, global). Colours come from CSS tokens so
   both themes work. */
const d3 = window.d3;
const css = (name) => `var(--${name})`;

function size(el, fallbackH) {
  const w = el.clientWidth || 600;
  return { w, h: fallbackH };
}

// ---------------------------------------------------------------------------
// Account: transfers over time (money in above zero, out below)
// ---------------------------------------------------------------------------
export function scatterTransfers(el, transfers) {
  el.innerHTML = "";
  if (!transfers.length) { el.innerHTML = '<p class="muted">No transfers.</p>'; return; }
  const { w, h } = size(el, 300);
  const m = { t: 10, r: 30, b: 30, l: 62 };
  const data = transfers.map((t) => ({ ...t, x: new Date(t.time), y: t.direction === "In" ? t.amount : -t.amount }));
  const x = d3.scaleTime().domain(d3.extent(data, (d) => d.x)).nice().range([m.l, w - m.r]);
  const ymax = d3.max(data, (d) => Math.abs(d.y)) || 1;
  const y = d3.scaleLinear().domain([-ymax, ymax]).nice().range([h - m.b, m.t]);
  const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${w} ${h}`).attr("width", "100%")
    .attr("role", "img").attr("aria-label", "Transfers over time: money in above the line, money out below");
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${h - m.b})`).call(d3.axisBottom(x).ticks(Math.min(6, w / 110)).tickSizeOuter(0));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`)
    .call(d3.axisLeft(y).ticks(6).tickFormat((v) => (v < 0 ? "−" : "") + "S$" + d3.format("~s")(Math.abs(v))).tickSize(-(w - m.l - m.r)))
    .call((g) => g.selectAll(".tick line").style("stroke", css("rule")).style("stroke-opacity", 0.5))
    .call((g) => g.select(".domain").remove());
  svg.append("line").attr("x1", m.l).attr("x2", w - m.r).attr("y1", y(0)).attr("y2", y(0)).style("stroke", css("muted"));
  const order = [...data].sort((a, b) => a.flagged - b.flagged);
  svg.append("g").selectAll("circle").data(order).join("circle")
    .attr("cx", (d) => x(d.x)).attr("cy", (d) => y(d.y)).attr("r", (d) => (d.flagged ? 5 : 3.5))
    .style("fill", (d) => (d.flagged ? css("critical") : css("faint")))
    .style("fill-opacity", (d) => (d.flagged ? 0.95 : 0.6))
    .style("stroke", (d) => (d.alert ? css("text") : "none")).style("stroke-width", 1.5)
    .append("title").text((d) => `${d.txn_id} · ${d.direction} S$${d3.format(",.0f")(d.amount)} · score ${d.score.toFixed(3)}`);
}

// ---------------------------------------------------------------------------
// Money flow: force-directed graph
// ---------------------------------------------------------------------------
export function flowGraph(el, graph, { onRecentre, onOpen, money }) {
  const tip = el.querySelector(".tip");
  el.querySelectorAll("svg").forEach((s) => s.remove());
  const w = el.clientWidth || 900, h = el.clientHeight || 560;
  if (!graph.links.length) {
    el.insertAdjacentHTML("beforeend", '<svg></svg>');
    const s = d3.select(el).select("svg").attr("viewBox", `0 0 ${w} ${h}`);
    s.append("text").attr("x", w / 2).attr("y", h / 2).attr("text-anchor", "middle").text("No transfers above the amount filter. Lower it to see this account's flows.");
    return;
  }
  const nodes = graph.nodes.map((n) => ({ ...n }));
  const links = graph.links.map((l) => ({ ...l }));
  const maxTotal = d3.max(links, (l) => l.total) || 1;
  const width = d3.scaleSqrt().domain([0, maxTotal]).range([1, 9]);
  const colour = (n) => css(n.kind === "external" ? "external" : n.kind);
  const radius = (n) => (n.center ? 16 : n.kind === "external" ? 6 : n.kind === "critical" ? 11 : 8);

  const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${w} ${h}`)
    .attr("role", "img").attr("aria-label", "Money flow graph around the selected account");
  const defs = svg.append("defs");
  for (const [id, c] of [["arrow-hot", "critical"], ["arrow", "faint"]]) {
    defs.append("marker").attr("id", id).attr("viewBox", "0 -5 10 10").attr("refX", 10).attr("refY", 0)
      .attr("markerUnits", "userSpaceOnUse").attr("markerWidth", 11).attr("markerHeight", 11).attr("orient", "auto")
      .append("path").attr("d", "M0,-5L10,0L0,5").style("fill", css(c));
  }
  const root = svg.append("g");
  svg.call(d3.zoom().scaleExtent([0.4, 3]).on("zoom", (e) => root.attr("transform", e.transform)));

  // Curve parallel edges (A→B and B→A) apart.
  const pairKey = (l) => [l.source, l.target].sort().join("|");
  const pairs = d3.group(links, pairKey);
  links.forEach((l) => { l.bend = pairs.get(pairKey(l)).length > 1 ? (l.source < l.target ? 1 : -1) : 0; });

  const link = root.append("g").selectAll("path").data(links).join("path")
    .style("fill", "none")
    .style("stroke", (l) => (l.flagged ? css("critical") : css("faint")))
    .style("stroke-opacity", (l) => (l.flagged ? 0.85 : 0.45))
    .style("stroke-width", (l) => width(l.total))
    .attr("marker-end", (l) => `url(#${l.flagged ? "arrow-hot" : "arrow"})`);

  const node = root.append("g").selectAll("g").data(nodes).join("g").style("cursor", "pointer");
  node.filter((n) => n.kind === "critical").append("circle").attr("r", (n) => radius(n) + 9)
    .style("fill", css("critical")).style("fill-opacity", 0.14);
  node.append((n) => document.createElementNS("http://www.w3.org/2000/svg", n.kind === "external" ? "rect" : "circle"))
    .each(function (n) {
      const r = radius(n);
      if (n.kind === "external") d3.select(this).attr("x", -r).attr("y", -r).attr("width", 2 * r).attr("height", 2 * r).attr("rx", 2);
      else d3.select(this).attr("r", r);
    })
    .style("fill", colour)
    .style("stroke", (n) => (n.center ? css("text") : css("panel"))).style("stroke-width", (n) => (n.center ? 2.5 : 1.5));
  node.append("text").attr("class", (n) => `node-label${n.center ? " center" : ""}`)
    .attr("text-anchor", "middle").attr("dy", (n) => radius(n) + 14)
    .text((n) => (n.label.length > 24 ? n.label.slice(0, 23) + "…" : n.label));

  const showTip = (event, html) => {
    tip.innerHTML = html; tip.style.display = "block";
    const box = el.getBoundingClientRect();
    tip.style.left = Math.min(event.clientX - box.left + 14, box.width - 290) + "px";
    tip.style.top = event.clientY - box.top + 14 + "px";
  };
  const hideTip = () => { tip.style.display = "none"; };
  node.on("mousemove", (e, n) => showTip(e, `<b>${n.label}</b><br>${n.label !== n.id ? n.id + "<br>" : ""}${n.kind === "external" ? "Outside our customer book" : `Score ${n.score}/10, ${n.kind}`}`))
    .on("mouseleave", hideTip)
    .on("click", (e, n) => { if (n.kind !== "external" && !n.center) onRecentre(n.id); })
    .on("dblclick", (e, n) => { if (n.kind !== "external") onOpen(n.id); });
  link.on("mousemove", (e, l) => showTip(e, `${l.source.id ?? l.source} → ${l.target.id ?? l.target}<br><b>${money(l.total)}</b> in ${l.n} transfer${l.n === 1 ? "" : "s"}<br>${l.flagged} scored 0.75+ · highest ${l.max_score.toFixed(3)}<br>${l.first} to ${l.last}`))
    .on("mouseleave", hideTip);

  const sim = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id((n) => n.id).distance((l) => (l.source.center || l.target.center ? 200 : 150)).strength(0.5))
    .force("charge", d3.forceManyBody().strength(-900))
    .force("center", d3.forceCenter(w / 2, h / 2))
    .force("collide", d3.forceCollide().radius((n) => radius(n) + 34))
    .force("x", d3.forceX(w / 2).strength(0.03)).force("y", d3.forceY(h / 2).strength(0.08));
  const centre = nodes.find((n) => n.center);
  if (centre) { centre.fx = w / 2; centre.fy = h / 2; }

  node.call(d3.drag()
    .on("start", (e, n) => { if (!e.active) sim.alphaTarget(0.2).restart(); n.fx = n.x; n.fy = n.y; })
    .on("drag", (e, n) => { n.fx = e.x; n.fy = e.y; })
    .on("end", (e, n) => { if (!e.active) sim.alphaTarget(0); if (!n.center) { n.fx = null; n.fy = null; } }));

  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const draw = () => {
    link.attr("d", (l) => {
      const sx = l.source.x, sy = l.source.y, tx = l.target.x, ty = l.target.y;
      const dx = tx - sx, dy = ty - sy, dist = Math.hypot(dx, dy) || 1;
      const r = radius(l.target) + 4;
      const ex = tx - (dx / dist) * r, ey = ty - (dy / dist) * r;
      if (!l.bend) return `M${sx},${sy}L${ex},${ey}`;
      const mx = (sx + ex) / 2 - (dy / dist) * 26 * l.bend, my = (sy + ey) / 2 + (dx / dist) * 26 * l.bend;
      return `M${sx},${sy}Q${mx},${my} ${ex},${ey}`;
    });
    node.attr("transform", (n) => `translate(${n.x},${n.y})`);
  };
  sim.on("tick.bounds", () => nodes.forEach((n) => {
    n.x = Math.max(40, Math.min(w - 40, n.x)); n.y = Math.max(30, Math.min(h - 40, n.y));
  }));
  if (reduce) { sim.stop(); for (let i = 0; i < 300; i++) sim.tick(); draw(); }
  else sim.on("tick", draw);
}

// ---------------------------------------------------------------------------
// Model: precision / recall across thresholds
// ---------------------------------------------------------------------------
export function lineChart(el, rows) {
  el.innerHTML = "";
  const { w, h } = size(el, 220);
  const m = { t: 10, r: 14, b: 30, l: 44 };
  const x = d3.scalePoint().domain(rows.map((r) => String(r.threshold))).range([m.l, w - m.r]).padding(0.2);
  const y = d3.scaleLinear().domain([0, 1]).range([h - m.b, m.t]);
  const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${w} ${h}`).attr("width", "100%")
    .attr("role", "img").attr("aria-label", "Precision and recall at each threshold");
  svg.append("g").attr("class", "axis").attr("transform", `translate(0,${h - m.b})`).call(d3.axisBottom(x).tickSizeOuter(0));
  svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`)
    .call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".0%")).tickSize(-(w - m.l - m.r)))
    .call((g) => g.selectAll(".tick line").style("stroke", css("rule")).style("stroke-opacity", 0.5))
    .call((g) => g.select(".domain").remove());
  for (const [key, colour] of [["recall", "act"], ["precision", "watch"]]) {
    svg.append("path").datum(rows)
      .attr("d", d3.line().x((r) => x(String(r.threshold))).y((r) => y(r[key])))
      .style("fill", "none").style("stroke", css(colour)).style("stroke-width", 2.5);
    svg.append("g").selectAll("circle").data(rows).join("circle")
      .attr("cx", (r) => x(String(r.threshold))).attr("cy", (r) => y(r[key])).attr("r", 3.5).style("fill", css(colour))
      .append("title").text((r) => `${key} ${(r[key] * 100).toFixed(1)}% at ${r.threshold}`);
  }
}

// ---------------------------------------------------------------------------
// Model: feature importance bars
// ---------------------------------------------------------------------------
export function hbar(el, rows) {
  el.innerHTML = "";
  const rowH = 30;
  const { w } = size(el, 0);
  const h = rows.length * rowH + 10;
  const labelW = Math.min(250, w * 0.45);
  const x = d3.scaleLinear().domain([0, d3.max(rows, (r) => r.share) || 1]).range([0, w - labelW - 50]);
  const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${w} ${h}`).attr("width", "100%")
    .attr("role", "img").attr("aria-label", "Feature importance");
  const g = svg.selectAll("g").data(rows).join("g").attr("transform", (r, i) => `translate(0,${i * rowH + 4})`);
  g.append("text").attr("x", labelW - 10).attr("y", rowH / 2).attr("dy", "0.1em").attr("text-anchor", "end")
    .text((r) => r.label).style("fill", css("text"));
  g.append("rect").attr("x", labelW).attr("y", 5).attr("height", rowH - 12).attr("width", (r) => Math.max(2, x(r.share)))
    .attr("rx", 2).style("fill", css("act"));
  g.append("text").attr("x", (r) => labelW + x(r.share) + 6).attr("y", rowH / 2).attr("dy", "0.1em")
    .text((r) => `${(r.share * 100).toFixed(0)}%`);
}
