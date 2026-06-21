/* analysis.js — DOM-free analytics over the Elden Ring graph.
 *
 * Everything that depends on the user's chosen variables (which edge types and
 * node types are active, Louvain resolution, the centrality metric) is computed
 * here, in the browser, from docs/data/graph.json. This is what makes the site
 * genuinely interactive rather than a gallery of pre-rendered figures.
 *
 * Globals expected (loaded via <script> before this file):
 *   graphology         -> { Graph, UndirectedGraph, ... }
 *   graphologyLibrary  -> { communitiesLouvain, metrics, ... }
 */
(function (global) {
  "use strict";

  const Graph = (global.graphology && global.graphology.UndirectedGraph) || null;
  const lib = global.graphologyLibrary || null;

  const FACTION_COLORS = {
    Intelligence: "#4aa3ff",
    Faith: "#e8b23a",
    Hybrid: "#b06bd6",
    Neither: "#9a9488",
  };

  const TYPE_COLORS = {
    weapon: "#e06c4f", shield: "#c9913f", sorcery: "#4aa3ff",
    incantation: "#e8b23a", ash_of_war: "#9b7be0", talisman: "#46c2a8",
    armor: "#8a93a3", item: "#7f8c9b", npc: "#ef6ea8", boss: "#d6453f",
    creature: "#7bbf5a", location: "#5fc3d6",
  };

  // ---- color helpers ------------------------------------------------------ //
  function communityColor(idx) {
    // Even hue spacing on a golden-angle walk -> distinct, stable colors.
    const hue = (idx * 137.508) % 360;
    return `hsl(${hue.toFixed(0)}, 62%, 58%)`;
  }

  function sentimentColor(s) {
    // diverging red(-1) -> grey(0) -> green(+1)
    const neg = [217, 72, 79], mid = [120, 128, 140], pos = [63, 181, 104];
    const t = Math.max(-1, Math.min(1, s));
    const lerp = (a, b, u) => a.map((v, i) => Math.round(v + (b[i] - v) * u));
    const rgb = t < 0 ? lerp(mid, neg, -t) : lerp(mid, pos, t);
    return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
  }

  // ---- build the filtered analysis graph ---------------------------------- //
  // Returns a graphology UndirectedGraph containing only the active node types
  // and active edge types. Parallel edges of different types collapse to the
  // max weight (so shares_location strength is respected).
  function buildFilteredGraph(data, params) {
    const g = new Graph();
    const activeNodeTypes = params.nodeTypes;
    const nodeOk = (n) => activeNodeTypes.has(n.type);

    for (const n of data.nodes) {
      if (nodeOk(n)) g.addNode(n.id);
    }
    for (const e of data.edges) {
      if (!params.edgeTypes.has(e.type)) continue;
      if (!g.hasNode(e.s) || !g.hasNode(e.t) || e.s === e.t) continue;
      if (g.hasEdge(e.s, e.t)) {
        const w = g.getEdgeAttribute(e.s, e.t, "weight");
        if (e.w > w) g.setEdgeAttribute(e.s, e.t, "weight", e.w);
      } else {
        g.addEdge(e.s, e.t, { weight: e.w || 1 });
      }
    }
    return g;
  }

  // ---- centrality (defensive: library if present, else local) ------------- //
  function degreeMap(g) {
    const out = {};
    g.forEachNode((n) => { out[n] = g.degree(n); });
    return out;
  }

  function centrality(g, metric) {
    const c = lib && lib.metrics && lib.metrics.centrality;
    try {
      if (metric === "pagerank" && c && c.pagerank) {
        return c.pagerank(g, { getEdgeWeight: "weight" });
      }
      if (metric === "betweenness" && c && c.betweenness) {
        return c.betweenness(g, { getEdgeWeight: "weight" });
      }
    } catch (err) {
      console.warn("centrality fallback:", err && err.message);
    }
    return degreeMap(g); // degree is the universal, dependency-free fallback
  }

  // Seeded PRNG (mulberry32) so Louvain is reproducible: identical parameters
  // always yield identical communities, which matters for an academic result.
  function seededRng(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // ---- community detection (Louvain) -------------------------------------- //
  function communities(g, resolution) {
    if (g.order === 0) return { map: {}, count: 0, modularity: 0 };
    try {
      const det = lib.communitiesLouvain.detailed(g, {
        resolution: resolution,
        getEdgeWeight: "weight",
        rng: seededRng(42),
      });
      return { map: det.communities, count: det.count, modularity: det.modularity };
    } catch (err) {
      console.warn("louvain fallback to connected components:", err && err.message);
      // Fallback: connected components as communities.
      const map = {}; let c = 0;
      const seen = new Set();
      g.forEachNode((start) => {
        if (seen.has(start)) return;
        const stack = [start];
        while (stack.length) {
          const v = stack.pop();
          if (seen.has(v)) continue;
          seen.add(v); map[v] = c;
          g.forEachNeighbor(v, (nb) => { if (!seen.has(nb)) stack.push(nb); });
        }
        c += 1;
      });
      return { map, count: c, modularity: 0 };
    }
  }

  // ---- TF-IDF keywords per community -------------------------------------- //
  // tf = summed token counts over community members; idf = log(N / df).
  function communityKeywords(data, commMap, nodeById, topComms, perComm) {
    const byComm = new Map();           // comm -> Map(token -> tf)
    for (const id in commMap) {
      const node = nodeById.get(id);
      if (!node || !node.tok) continue;
      const c = commMap[id];
      if (!byComm.has(c)) byComm.set(c, new Map());
      const bag = byComm.get(c);
      for (const tok in node.tok) bag.set(tok, (bag.get(tok) || 0) + node.tok[tok]);
    }
    const N = byComm.size || 1;
    const df = new Map();
    for (const bag of byComm.values()) {
      for (const tok of bag.keys()) df.set(tok, (df.get(tok) || 0) + 1);
    }
    const out = {};
    for (const c of topComms) {
      const bag = byComm.get(c);
      if (!bag) { out[c] = []; continue; }
      let total = 0; bag.forEach((v) => (total += v));
      const scored = [];
      bag.forEach((tf, tok) => {
        const idf = Math.log(N / (df.get(tok) || 1)) + 1e-6;
        scored.push([tok, (tf / total) * idf]);
      });
      scored.sort((a, b) => b[1] - a[1]);
      out[c] = scored.slice(0, perComm).map((x) => x[0]);
    }
    return out;
  }

  // dominant non-Neither faction within each community
  function communityFactions(commMap, nodeById, comms) {
    const out = {};
    for (const c of comms) out[c] = { Intelligence: 0, Faith: 0, Hybrid: 0 };
    for (const id in commMap) {
      const node = nodeById.get(id);
      const c = commMap[id];
      if (!node || !(c in out)) continue;
      if (node.faction in out[c]) out[c][node.faction] += 1;
    }
    const label = {};
    for (const c of comms) {
      const f = out[c];
      const tot = f.Intelligence + f.Faith + f.Hybrid;
      if (!tot) { label[c] = "Mixed"; continue; }
      if (f.Intelligence >= f.Faith && f.Intelligence >= f.Hybrid) label[c] = "Int-leaning";
      else if (f.Faith >= f.Hybrid) label[c] = "Faith-leaning";
      else label[c] = "Hybrid-leaning";
    }
    return label;
  }

  // Custom Sigma hover/highlight label: a dark, gold-bordered pill with light
  // text, instead of Sigma's default WHITE box (which made the near-white label
  // invisible on highlight). Shared by both the Network and Realm-of-Shadow graphs.
  function hoverLabel(context, data, settings) {
    const size = settings.labelSize || 12;
    const font = settings.labelFont || "sans-serif";
    const weight = settings.labelWeight || "normal";
    context.font = `${weight} ${size}px ${font}`;
    const label = data.label;
    const pad = 6;
    const tw = label ? context.measureText(label).width : 0;
    const bw = data.size * 2 + 4 + (label ? tw + pad * 2 : 0);
    const bh = Math.max(size + pad, data.size * 2 + 4);
    const bx = data.x - data.size - 2;
    const by = data.y - bh / 2;

    context.fillStyle = "rgba(15,13,9,0.94)";
    context.strokeStyle = "#c9a86a";
    context.lineWidth = 1;
    context.beginPath();
    if (context.roundRect) context.roundRect(bx, by, bw, bh, 5);
    else context.rect(bx, by, bw, bh);
    context.closePath();
    context.fill();
    context.stroke();

    // redraw the node disc on top of the pill
    context.fillStyle = data.color || "#9a9488";
    context.beginPath();
    context.arc(data.x, data.y, data.size, 0, Math.PI * 2);
    context.closePath();
    context.fill();

    if (label) {
      context.fillStyle = "#ece5d2";
      context.textAlign = "left";
      context.textBaseline = "middle";
      context.fillText(label, data.x + data.size + pad, data.y);
    }
  }

  global.ERA = global.ERA || {};
  global.ERA.analysis = {
    FACTION_COLORS, TYPE_COLORS, communityColor, sentimentColor,
    buildFilteredGraph, centrality, communities, communityKeywords,
    communityFactions, degreeMap, hoverLabel,
  };
})(window);
