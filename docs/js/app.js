/* app.js — UI, Sigma rendering, charts, and control wiring.
 * Depends on globals: graphology, graphologyLibrary, Sigma, Chart, ERA.analysis
 */
(function () {
  "use strict";

  const A = window.ERA.analysis;
  const Graph = window.graphology.UndirectedGraph;

  const TYPE_LABELS = {
    weapon: "Weapons", shield: "Shields", sorcery: "Sorceries",
    incantation: "Incantations", ash_of_war: "Ashes of War", talisman: "Talismans",
    armor: "Armor", item: "Items", npc: "NPCs", boss: "Bosses",
    creature: "Creatures", location: "Locations",
  };
  const EDGE_LABELS = {
    mentions: "Mentions (text)", located_in: "Located in",
    drops: "Drops", shares_location: "Shares location",
  };
  const PRESETS = {
    characters: ["npc", "boss", "creature"],
    arsenal: ["weapon", "shield", "sorcery", "incantation", "ash_of_war"],
  };

  const STATE = {
    data: null,
    nodeById: new Map(),
    renderer: null,
    MG: null,
    params: {
      edgeTypes: new Set(),
      nodeTypes: new Set(),
      colorBy: "faction",
      sizeBy: "pagerank",
      resolution: 1.0,
      minDegree: 1,
      factionHighlight: "all",
    },
    visibleNodes: new Set(),
    nodeColor: new Map(),
    nodeSize: new Map(),
    community: {},
    centrality: {},
    commLabel: {},
    selected: null,
    hovered: null,
    hoverSet: null,
  };

  let factionChart = null, sentimentChart = null;

  // ===================================================================== //
  // Boot
  // ===================================================================== //
  async function boot() {
    // Prefer inlined data (window.ERA_DATA) so the page works offline / via
    // file:// with no server; fall back to fetch for modular dev serving.
    let data = window.ERA_DATA;
    if (!data) {
      const resp = await fetch("data/graph.json");
      if (!resp.ok) throw new Error("Could not load data/graph.json");
      data = await resp.json();
    }
    STATE.data = data;
    data.nodes.forEach((n) => STATE.nodeById.set(n.id, n));

    STATE.params.edgeTypes = new Set(Object.keys(data.meta.edge_types));
    STATE.params.nodeTypes = new Set(Object.keys(data.meta.node_types));

    const gen = (data.meta.generated_at || "").slice(0, 10);
    document.getElementById("meta").innerHTML =
      `<span><b>${data.meta.node_count}</b> nodes · <b>${data.meta.edge_count}</b> edges</span>` +
      `<span>source: ${data.meta.source}</span>` +
      (gen ? `<span>built ${gen}</span>` : "");

    buildControls();
    // Sync slider DOM to defaults (guards against browser form-restore on reload).
    setRange("resolution", STATE.params.resolution, STATE.params.resolution.toFixed(1));
    setRange("min-degree", STATE.params.minDegree, String(STATE.params.minDegree));
    buildMasterGraph();
    initSigma();
    initCharts();
    recompute();
    document.getElementById("loading").style.display = "none";

    // shared API for the Story / Realm-of-Shadow views
    window.ERA.app = {
      data: STATE.data, nodeById: STATE.nodeById, analysis: A,
      typeLabels: TYPE_LABELS, edgeLabels: EDGE_LABELS,
      focusInNetwork, switchView,
    };
    if (window.ERA.story) window.ERA.story.init();
    if (window.ERA.dlc) window.ERA.dlc.init();
    initTabs();
  }

  function initTabs() {
    const tabs = document.querySelectorAll("#tabs .tab");
    tabs.forEach((t) => t.addEventListener("click", () => switchView(t.dataset.view)));
  }

  function switchView(name) {
    document.querySelectorAll("#tabs .tab").forEach((t) =>
      t.classList.toggle("active", t.dataset.view === name));
    document.querySelectorAll(".view").forEach((v) => v.classList.add("hidden"));
    document.getElementById("view-" + name).classList.remove("hidden");
    if (name === "network" && STATE.renderer) { STATE.renderer.refresh(); }
    if (name === "dlc" && window.ERA.dlc) window.ERA.dlc.activate();
  }

  function focusInNetwork(id) {
    const n = STATE.nodeById.get(id);
    if (!n) return;
    if (!STATE.params.nodeTypes.has(n.type)) {
      STATE.params.nodeTypes.add(n.type); syncNodeTypeBox(n.type, true); recompute();
    }
    switchView("network");
    setTimeout(() => selectNode(id, true), 60);
  }

  // ===================================================================== //
  // Master graph + Sigma
  // ===================================================================== //
  function buildMasterGraph() {
    const g = new Graph();
    for (const n of STATE.data.nodes) {
      g.addNode(n.id, { x: n.x, y: n.y, size: 4, label: n.name });
    }
    for (const e of STATE.data.edges) {
      if (g.hasEdge(e.s, e.t)) continue; // sigma can't hold parallels here
      if (e.s === e.t || !g.hasNode(e.s) || !g.hasNode(e.t)) continue;
      g.addEdge(e.s, e.t, { etype: e.type, s: e.s, t: e.t, weight: e.w || 1 });
    }
    STATE.MG = g;
  }

  function nodeVisible(id) {
    if (!STATE.visibleNodes.has(id)) return false;
    return true;
  }

  function initSigma() {
    const container = document.getElementById("graph");
    STATE.renderer = new Sigma(STATE.MG, container, {
      renderLabels: true,
      labelRenderedSizeThreshold: 16,
      labelColor: { color: "#dfe3ea" },
      labelFont: "Inter, system-ui, sans-serif",
      labelSize: 12,
      defaultNodeColor: "#9a9488",
      defaultEdgeColor: "#262b33",
      defaultDrawNodeHover: A.hoverLabel,
      zIndex: true,
      nodeReducer: (id, data) => {
        const res = { ...data };
        if (!nodeVisible(id)) { res.hidden = true; return res; }
        res.color = STATE.nodeColor.get(id) || "#9a9488";
        res.size = STATE.nodeSize.get(id) || 3;
        const hov = STATE.hoverSet;
        if (hov && !hov.has(id)) { res.color = "#2c3039"; res.label = ""; res.zIndex = 0; }
        else if (hov) { res.zIndex = 2; res.forceLabel = true; }
        if (STATE.selected === id) { res.highlighted = true; res.forceLabel = true; res.zIndex = 3; }
        return res;
      },
      edgeReducer: (id, data) => {
        const res = { ...data };
        if (!STATE.params.edgeTypes.has(data.etype) ||
            !nodeVisible(data.s) || !nodeVisible(data.t)) {
          res.hidden = true; return res;
        }
        const hov = STATE.hoverSet;
        if (hov && !(hov.has(data.s) && hov.has(data.t))) { res.hidden = true; return res; }
        res.color = hov ? "#6b7686" : "#2b313b";
        res.size = data.etype === "shares_location" ? Math.min(1 + data.weight * 0.3, 3) : 0.6;
        return res;
      },
    });

    STATE.renderer.on("clickNode", ({ node }) => selectNode(node, false));
    STATE.renderer.on("enterNode", ({ node }) => setHover(node));
    STATE.renderer.on("leaveNode", () => setHover(null));
    STATE.renderer.on("clickStage", () => { STATE.selected = null; closeDetail(); STATE.renderer.refresh(); });
  }

  function setHover(node) {
    STATE.hovered = node;
    if (!node) { STATE.hoverSet = null; STATE.renderer.refresh(); return; }
    const set = new Set([node]);
    STATE.MG.forEachNeighbor(node, (nb, attr) => {
      if (STATE.params.edgeTypes.has(attr.etype) && nodeVisible(nb)) set.add(nb);
    });
    // also include neighbors reachable via edges where 'node' is the other end
    STATE.MG.forEachEdge(node, (e, attr, s, t) => {
      if (!STATE.params.edgeTypes.has(attr.etype)) return;
      const other = s === node ? t : s;
      if (nodeVisible(other)) set.add(other);
    });
    STATE.hoverSet = set;
    STATE.renderer.refresh();
  }

  // ===================================================================== //
  // Recompute pipeline (the interactive core)
  // ===================================================================== //
  function recompute() {
    const p = STATE.params;
    const fg = A.buildFilteredGraph(STATE.data, p);

    const metric = p.sizeBy === "uniform" ? "degree" : p.sizeBy;
    const cent = A.centrality(fg, metric);
    const comm = A.communities(fg, p.resolution);
    STATE.centrality = cent;
    STATE.community = comm.map;

    // visible = active-type nodes whose degree in the filtered graph >= minDegree
    const effMin = Math.max(p.minDegree, 0);
    const vis = new Set();
    fg.forEachNode((id) => { if (fg.degree(id) >= effMin) vis.add(id); });
    STATE.visibleNodes = vis;

    // colors + sizes
    const centVals = [...vis].map((id) => cent[id] || 0);
    const cMax = Math.max(...centVals, 1e-9);
    STATE.nodeColor = new Map();
    STATE.nodeSize = new Map();
    for (const id of vis) {
      const node = STATE.nodeById.get(id);
      STATE.nodeColor.set(id, colorFor(node, comm.map[id]));
      const base = p.sizeBy === "uniform" ? 4 : 2 + 16 * Math.sqrt((cent[id] || 0) / cMax);
      STATE.nodeSize.set(id, base);
    }
    applyFactionHighlight();

    // ---- panels ---- //
    const visEdges = countVisibleEdges(fg, vis);
    const commCount = new Set([...vis].map((id) => comm.map[id])).size;
    setStat("stat-nodes", vis.size);
    setStat("stat-edges", visEdges);
    setStat("stat-comms", commCount);
    setStat("stat-mod", comm.modularity ? comm.modularity.toFixed(3) : "—");

    updateCharts(vis);
    updateCommunityPanel(vis, comm.map);
    updateHubs(vis, cent);
    updateLegend();

    if (STATE.renderer) STATE.renderer.refresh();
  }

  function colorFor(node, comm) {
    switch (STATE.params.colorBy) {
      case "type": return A.TYPE_COLORS[node.type] || "#9a9488";
      case "community": return comm === undefined ? "#9a9488" : A.communityColor(comm);
      case "sentiment": return A.sentimentColor(node.sent || 0);
      default: return A.FACTION_COLORS[node.faction] || "#9a9488";
    }
  }

  function applyFactionHighlight() {
    const hl = STATE.params.factionHighlight;
    if (hl === "all") return;
    for (const id of STATE.visibleNodes) {
      const node = STATE.nodeById.get(id);
      if (node.faction !== hl) {
        STATE.nodeColor.set(id, "#2c3039");
        STATE.nodeSize.set(id, Math.min(STATE.nodeSize.get(id), 3));
      }
    }
  }

  function countVisibleEdges(fg, vis) {
    let n = 0;
    fg.forEachEdge((e, a, s, t) => { if (vis.has(s) && vis.has(t)) n += 1; });
    return n;
  }

  // ===================================================================== //
  // Charts
  // ===================================================================== //
  function initCharts() {
    const cFac = document.getElementById("chart-faction").getContext("2d");
    factionChart = new Chart(cFac, {
      type: "doughnut",
      data: { labels: [], datasets: [{ data: [], backgroundColor: [] }] },
      options: { maintainAspectRatio: false,
        plugins: { legend: { position: "bottom", labels: { color: "#cfd4dc", boxWidth: 12, font: { size: 11 } } } },
        cutout: "55%" },
    });
    const cSent = document.getElementById("chart-sentiment").getContext("2d");
    sentimentChart = new Chart(cSent, {
      type: "bar",
      data: { labels: [], datasets: [{ label: "Mean sentiment", data: [], backgroundColor: [] }] },
      options: {
        indexAxis: "y",
        maintainAspectRatio: false,
        scales: {
          x: { min: -1, max: 1, grid: { color: "#262b33" }, ticks: { color: "#9aa3b0", font: { size: 10 } } },
          y: { grid: { display: false }, ticks: { color: "#cfd4dc", font: { size: 10 } } },
        },
        plugins: { legend: { display: false } },
      },
    });
  }

  function updateCharts(vis) {
    // faction doughnut
    const fc = { Intelligence: 0, Faith: 0, Hybrid: 0, Neither: 0 };
    for (const id of vis) fc[STATE.nodeById.get(id).faction] += 1;
    factionChart.data.labels = Object.keys(fc);
    factionChart.data.datasets[0].data = Object.values(fc);
    factionChart.data.datasets[0].backgroundColor = Object.keys(fc).map((k) => A.FACTION_COLORS[k]);
    factionChart.update();

    // mean sentiment by node type
    const agg = {};
    for (const id of vis) {
      const n = STATE.nodeById.get(id);
      (agg[n.type] = agg[n.type] || []).push(n.sent || 0);
    }
    const rows = Object.entries(agg)
      .map(([t, arr]) => [t, arr.reduce((a, b) => a + b, 0) / arr.length])
      .sort((a, b) => a[1] - b[1]);
    sentimentChart.data.labels = rows.map((r) => TYPE_LABELS[r[0]] || r[0]);
    sentimentChart.data.datasets[0].data = rows.map((r) => +r[1].toFixed(3));
    sentimentChart.data.datasets[0].backgroundColor = rows.map((r) => A.sentimentColor(r[1]));
    sentimentChart.update();
  }

  // ===================================================================== //
  // Community + hubs panels
  // ===================================================================== //
  function updateCommunityPanel(vis, commMap) {
    const sizes = {};
    for (const id of vis) { const c = commMap[id]; sizes[c] = (sizes[c] || 0) + 1; }
    const top = Object.entries(sizes).sort((a, b) => b[1] - a[1]).slice(0, 8).map((x) => +x[0]);
    const visMap = {};
    for (const id of vis) visMap[id] = commMap[id];
    const kw = A.communityKeywords(STATE.data, visMap, STATE.nodeById, top, 5);
    const fl = A.communityFactions(visMap, STATE.nodeById, top);
    STATE.commLabel = fl;

    const el = document.getElementById("community-list");
    el.innerHTML = "";
    for (const c of top) {
      const div = document.createElement("div");
      div.className = "comm-item";
      const swatch = STATE.params.colorBy === "community"
        ? A.communityColor(c)
        : "#3a4150";
      div.innerHTML =
        `<div class="comm-head"><span class="swatch" style="background:${swatch}"></span>` +
        `<b>Community ${c}</b><span class="muted">${sizes[c]} nodes · ${fl[c]}</span></div>` +
        `<div class="chips">${(kw[c] || []).map((w) => `<span class="chip">${w}</span>`).join("")}</div>`;
      el.appendChild(div);
    }
  }

  function updateHubs(vis, cent) {
    const rows = [...vis].map((id) => [id, cent[id] || 0])
      .sort((a, b) => b[1] - a[1]).slice(0, 12);
    const tbody = document.getElementById("hubs-body");
    tbody.innerHTML = "";
    for (const [id, score] of rows) {
      const n = STATE.nodeById.get(id);
      const tr = document.createElement("tr");
      tr.innerHTML =
        `<td class="hub-name">${n.name}</td>` +
        `<td><span class="badge" style="background:${A.TYPE_COLORS[n.type]}22;color:${A.TYPE_COLORS[n.type]}">${TYPE_LABELS[n.type] || n.type}</span></td>` +
        `<td>${fmtScore(score)}</td>`;
      tr.onclick = () => selectNode(id, true);
      tbody.appendChild(tr);
    }
  }

  function fmtScore(s) { return s >= 1 ? s.toFixed(0) : s.toFixed(4); }

  // ===================================================================== //
  // Node detail + selection
  // ===================================================================== //
  function selectNode(id, focus) {
    STATE.selected = id;
    const n = STATE.nodeById.get(id);
    const panel = document.getElementById("detail");
    panel.classList.add("open");

    const neighbors = collectNeighbors(id);
    const neighHtml = Object.entries(neighbors).map(([etype, list]) =>
      `<div class="nb-group"><div class="nb-type">${EDGE_LABELS[etype] || etype} (${list.length})</div>` +
      list.slice(0, 12).map((nb) =>
        `<span class="nb-link" data-id="${nb.id}">${nb.name}</span>`).join("") +
      `</div>`).join("") || `<div class="muted">No active connections.</div>`;

    const ia = Math.round((n.ia || 0) * 100), fa = Math.round((n.fa || 0) * 100);
    panel.querySelector(".detail-body").innerHTML =
      `<h3>${n.name}</h3>` +
      `<div class="tags"><span class="badge" style="background:${A.TYPE_COLORS[n.type]}22;color:${A.TYPE_COLORS[n.type]}">${TYPE_LABELS[n.type] || n.type}</span>` +
      `<span class="badge" style="background:${A.FACTION_COLORS[n.faction]}22;color:${A.FACTION_COLORS[n.faction]}">${n.faction}</span>` +
      (n.merchant ? `<span class="badge merch">merchant</span>` : "") +
      (n.fate ? `<span class="badge fate">${n.fate.replace("_", " ")}</span>` : "") + `</div>` +
      `<div class="affinity"><div class="aff-row"><span>INT</span><div class="bar"><i style="width:${ia}%;background:${A.FACTION_COLORS.Intelligence}"></i></div><span>${ia}</span></div>` +
      `<div class="aff-row"><span>FTH</span><div class="bar"><i style="width:${fa}%;background:${A.FACTION_COLORS.Faith}"></i></div><span>${fa}</span></div></div>` +
      `<div class="metrics"><span>Sentiment <b style="color:${A.sentimentColor(n.sent)}">${(n.sent || 0).toFixed(3)}</b></span>` +
      `<span>Community <b>${STATE.community[id] ?? "—"}</b></span>` +
      `<span>${STATE.params.sizeBy} <b>${fmtScore(STATE.centrality[id] || 0)}</b></span></div>` +
      (n.snippet ? `<p class="snippet">${n.snippet}${n.snippet.length >= 240 ? "…" : ""}</p>` : "") +
      `<div class="neighbors">${neighHtml}</div>`;

    panel.querySelectorAll(".nb-link").forEach((el) =>
      el.addEventListener("click", () => selectNode(el.dataset.id, true)));

    if (focus) focusNode(id);
    STATE.renderer.refresh();
  }

  function collectNeighbors(id) {
    const groups = {};
    STATE.MG.forEachEdge(id, (e, attr, s, t) => {
      if (!STATE.params.edgeTypes.has(attr.etype)) return;
      const other = s === id ? t : s;
      if (!nodeVisible(other)) return;
      (groups[attr.etype] = groups[attr.etype] || []).push(STATE.nodeById.get(other));
    });
    return groups;
  }

  function focusNode(id) {
    const disp = STATE.renderer.getNodeDisplayData(id);
    if (!disp) return;
    STATE.renderer.getCamera().animate({ x: disp.x, y: disp.y, ratio: 0.35 }, { duration: 500 });
  }

  function closeDetail() { document.getElementById("detail").classList.remove("open"); }

  // ===================================================================== //
  // Controls
  // ===================================================================== //
  function buildControls() {
    const meta = STATE.data.meta;

    const etWrap = document.getElementById("edge-types");
    Object.keys(meta.edge_types).forEach((t) => {
      etWrap.appendChild(checkbox(`et-${t}`, `${EDGE_LABELS[t] || t}`, meta.edge_types[t], true, (on) => {
        on ? STATE.params.edgeTypes.add(t) : STATE.params.edgeTypes.delete(t);
        recompute();
      }));
    });

    const ntWrap = document.getElementById("node-types");
    Object.keys(meta.node_types).forEach((t) => {
      ntWrap.appendChild(checkbox(`nt-${t}`, TYPE_LABELS[t] || t, meta.node_types[t], true, (on) => {
        on ? STATE.params.nodeTypes.add(t) : STATE.params.nodeTypes.delete(t);
        recompute();
      }));
    });

    // presets
    document.getElementById("preset-chars").onclick = () => applyNodePreset(PRESETS.characters);
    document.getElementById("preset-arsenal").onclick = () => applyNodePreset(PRESETS.arsenal);
    document.getElementById("preset-all").onclick = () => applyNodePreset(Object.keys(meta.node_types));

    bindSelect("color-by", (v) => { STATE.params.colorBy = v; recompute(); });
    bindSelect("size-by", (v) => { STATE.params.sizeBy = v; recompute(); });
    bindSelect("faction-highlight", (v) => { STATE.params.factionHighlight = v; recompute(); });

    bindRange("resolution", (v) => {
      STATE.params.resolution = +v;
      document.getElementById("resolution-val").textContent = (+v).toFixed(1);
    });
    bindRange("min-degree", (v) => {
      STATE.params.minDegree = +v;
      document.getElementById("min-degree-val").textContent = v;
    });

    document.getElementById("reset-btn").onclick = resetView;
    document.getElementById("detail-close").onclick = () => {
      closeDetail(); STATE.selected = null; STATE.renderer.refresh();
    };

    // search
    const search = document.getElementById("search");
    const results = document.getElementById("search-results");
    search.addEventListener("input", () => {
      const q = search.value.trim().toLowerCase();
      results.innerHTML = "";
      if (q.length < 2) return;
      const hits = STATE.data.nodes
        .filter((n) => n.name.toLowerCase().includes(q)).slice(0, 8);
      hits.forEach((n) => {
        const d = document.createElement("div");
        d.className = "search-hit";
        d.textContent = `${n.name} · ${TYPE_LABELS[n.type] || n.type}`;
        d.onclick = () => {
          if (!STATE.params.nodeTypes.has(n.type)) { STATE.params.nodeTypes.add(n.type); syncNodeTypeBox(n.type, true); recompute(); }
          selectNode(n.id, true); results.innerHTML = ""; search.value = n.name;
        };
        results.appendChild(d);
      });
    });
  }

  function applyNodePreset(types) {
    const set = new Set(types);
    STATE.params.nodeTypes = set;
    Object.keys(STATE.data.meta.node_types).forEach((t) => syncNodeTypeBox(t, set.has(t)));
    recompute();
  }
  function syncNodeTypeBox(t, on) { const el = document.getElementById(`nt-${t}`); if (el) el.checked = on; }

  function resetView() {
    STATE.params.edgeTypes = new Set(Object.keys(STATE.data.meta.edge_types));
    STATE.params.nodeTypes = new Set(Object.keys(STATE.data.meta.node_types));
    STATE.params.colorBy = "faction"; STATE.params.sizeBy = "pagerank";
    STATE.params.resolution = 1.0; STATE.params.minDegree = 1;
    STATE.params.factionHighlight = "all";
    document.querySelectorAll("#edge-types input,#node-types input").forEach((c) => (c.checked = true));
    setSelect("color-by", "faction"); setSelect("size-by", "pagerank"); setSelect("faction-highlight", "all");
    setRange("resolution", 1.0, "1.0"); setRange("min-degree", 1, "1");
    closeDetail(); STATE.selected = null;
    recompute();
    STATE.renderer.getCamera().animatedReset();
  }

  // ---- small DOM helpers -------------------------------------------------- //
  function checkbox(id, label, count, checked, onChange) {
    const wrap = document.createElement("label");
    wrap.className = "chk";
    wrap.innerHTML = `<input type="checkbox" id="${id}" ${checked ? "checked" : ""}>` +
      `<span>${label}</span><span class="count">${count}</span>`;
    wrap.querySelector("input").addEventListener("change", (e) => onChange(e.target.checked));
    return wrap;
  }
  function bindSelect(id, fn) { document.getElementById(id).addEventListener("change", (e) => fn(e.target.value)); }
  function setSelect(id, v) { document.getElementById(id).value = v; }
  function bindRange(id, fn) {
    const el = document.getElementById(id);
    el.addEventListener("input", (e) => fn(e.target.value));
    el.addEventListener("change", () => recompute());
  }
  function setRange(id, v, label) {
    document.getElementById(id).value = v;
    document.getElementById(`${id}-val`).textContent = label;
  }
  function setStat(id, v) { document.getElementById(id).textContent = v; }

  function updateLegend() {
    const el = document.getElementById("legend");
    const mode = STATE.params.colorBy;
    let items = [];
    if (mode === "faction") items = Object.entries(A.FACTION_COLORS);
    else if (mode === "type") items = Object.keys(STATE.data.meta.node_types).map((t) => [TYPE_LABELS[t] || t, A.TYPE_COLORS[t]]);
    else if (mode === "sentiment") items = [["negative", A.sentimentColor(-1)], ["neutral", A.sentimentColor(0)], ["positive", A.sentimentColor(1)]];
    else items = [["color = community", "#888"]];
    el.innerHTML = items.map(([k, c]) =>
      `<span class="leg"><span class="swatch" style="background:${c}"></span>${k}</span>`).join("");
  }

  window.addEventListener("DOMContentLoaded", () => {
    boot().catch((err) => {
      console.error(err);
      const l = document.getElementById("loading");
      if (l) l.textContent = "Failed to load: " + err.message;
    });
  });
})();
