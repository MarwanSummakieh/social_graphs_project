/* dlc.js — "Realm of Shadow": a curated, motive-focused map of Shadow of the
 * Erdtree. The fan API has no DLC data, so this view is built from a hand-
 * authored dataset (window.ERA_DATA.dlc) rendered through the same engine.
 */
(function () {
  "use strict";

  const GROUP_COLORS = {
    "Miquella's Circle": "#e8c170",
    "Marika's Legacy": "#c0392b",
    "The Hornsent": "#9b7be0",
    "Fingers & Outer Gods": "#5fc3d6",
    "Dragons": "#e07b39",
    "Frenzy & Abyss": "#d9d055",
  };

  let data, byId, renderer, graph, started = false, selected = null, adjacency = new Map();

  function init() {
    data = window.ERA_DATA && window.ERA_DATA.dlc;
    if (!data) return;
    byId = new Map(data.nodes.map((n) => [n.id, n]));
    for (const e of data.edges) {
      addAdj(e.s, { id: e.t, rel: e.rel, dir: "out" });
      addAdj(e.t, { id: e.s, rel: e.rel, dir: "in" });
    }
    renderIntro();
    renderLegend();
    renderList();
  }
  function addAdj(a, rec) { if (!adjacency.has(a)) adjacency.set(a, []); adjacency.get(a).push(rec); }

  function renderIntro() {
    document.getElementById("dlc-intro").innerHTML =
      `${data.meta.note} <b>${data.meta.node_count}</b> figures, <b>${data.meta.edge_count}</b> bonds. ` +
      `Click a figure to read its motive.`;
  }

  function renderLegend() {
    document.getElementById("dlc-legend").innerHTML = data.meta.groups.map((g) =>
      `<span class="leg"><span class="swatch" style="background:${GROUP_COLORS[g] || "#888"}"></span>${g}</span>`).join("");
  }

  // Sigma must be created when the container is visible (has size), so defer.
  function activate() {
    if (started) { renderer.refresh(); return; }
    started = true;
    const G = window.graphology.DirectedGraph;
    graph = new G();
    const deg = {};
    for (const e of data.edges) { deg[e.s] = (deg[e.s] || 0) + 1; deg[e.t] = (deg[e.t] || 0) + 1; }
    for (const n of data.nodes) {
      graph.addNode(n.id, {
        x: n.x, y: n.y, label: n.name,
        size: 6 + 2.2 * Math.sqrt(deg[n.id] || 1),
        color: GROUP_COLORS[n.group] || "#999",
      });
    }
    for (const e of data.edges) {
      if (!graph.hasEdge(e.s, e.t)) graph.addEdge(e.s, e.t, { label: e.rel, size: 1.4 });
    }

    renderer = new Sigma(graph, document.getElementById("dlc-graph"), {
      renderLabels: true,
      renderEdgeLabels: true,
      labelRenderedSizeThreshold: 0,
      defaultEdgeType: "arrow",
      labelColor: { color: "#efe7d2" },
      edgeLabelColor: { color: "#9a8e6f" },
      labelFont: "Georgia, serif",
      labelSize: 13,
      edgeLabelSize: 11,
      defaultEdgeColor: "#4a4330",
      defaultDrawNodeHover: window.ERA.analysis.hoverLabel,
      nodeReducer: (id, d) => {
        const r = { ...d };
        if (selected) {
          const near = id === selected || (adjacency.get(selected) || []).some((x) => x.id === id);
          if (!near) { r.color = "#3a3730"; r.label = ""; }
          if (id === selected) r.highlighted = true;
        }
        return r;
      },
      edgeReducer: (id, d) => {
        const r = { ...d };
        if (selected) {
          const [s, t] = graph.extremities(id);
          if (s !== selected && t !== selected) { r.hidden = true; }
          else { r.color = "#c9a86a"; r.size = 2; }
        }
        return r;
      },
    });
    renderer.on("clickNode", ({ node }) => showFigure(node, true));
    renderer.on("clickStage", () => { selected = null; renderer.refresh(); });
  }

  function renderList() {
    const wrap = document.getElementById("dlc-list");
    wrap.innerHTML = "";
    for (const g of data.meta.groups) {
      const members = data.nodes.filter((n) => n.group === g);
      if (!members.length) continue;
      const block = document.createElement("div");
      block.className = "dlc-group";
      block.innerHTML = `<div class="dlc-grouphead"><span class="swatch" style="background:${GROUP_COLORS[g]}"></span>${g}</div>`;
      for (const n of members) {
        const b = document.createElement("button");
        b.className = "dlc-figure";
        b.textContent = n.name;
        b.onclick = () => showFigure(n.id, true);
        block.appendChild(b);
      }
      wrap.appendChild(block);
    }
  }

  function showFigure(id, focus) {
    selected = id;
    const n = byId.get(id);
    const rels = (adjacency.get(id) || []).map((r) => {
      const o = byId.get(r.id);
      const arrow = r.dir === "out" ? "→" : "←";
      return `<div class="rel"><span class="rel-label">${r.rel}</span> ${arrow}
        <b class="link" data-id="${r.id}">${o ? o.name : r.id}</b></div>`;
    }).join("");

    document.getElementById("dlc-detail").innerHTML =
      `<div class="dlc-card">
        <div class="dlc-card-head">
          <span class="swatch" style="background:${GROUP_COLORS[n.group]}"></span>
          <h3>${n.name}</h3>
        </div>
        <div class="muted dlc-group-tag">${n.group}</div>
        <div class="motive"><div class="motive-label">Motive</div><p>${n.motive}</p></div>
        ${n.lore ? `<p class="dlc-lore">${n.lore}</p>` : ""}
        ${rels ? `<div class="rels"><div class="motive-label">Bonds</div>${rels}</div>` : ""}
      </div>`;

    document.querySelectorAll("#dlc-detail .link").forEach((el) =>
      el.addEventListener("click", () => showFigure(el.dataset.id, true)));

    if (renderer) {
      renderer.refresh();
      if (focus) {
        const disp = renderer.getNodeDisplayData(id);
        if (disp) renderer.getCamera().animate({ x: disp.x, y: disp.y, ratio: 0.55 }, { duration: 400 });
      }
    }
  }

  window.ERA = window.ERA || {};
  window.ERA.dlc = { init, activate };
})();
