/* story.js — "The Story": narrative dossiers reconstructed from the data.
 *
 * For a chosen character/place we surface (a) its own lore, (b) the "echoes" —
 * every item/location whose description NAMES it (incoming `mentions` edges),
 * which is literally the world telling that character's story through item
 * text, (c) its structural bonds (drops / location / co-location), (d) lore
 * events detected by keyword, and (e) the emotional tone via sentiment.
 */
(function () {
  "use strict";

  // Curated lore events (keyword-detected in the assembled text).
  const EVENTS = [
    ["The Shattering", ["shattering", "shattered"], "The war that broke the Elden Ring among Marika's demigod children."],
    ["Night of the Black Knives", ["black knife", "night of the black", "godwyn"], "The assassination that slew Godwyn's soul with stolen Destined Death."],
    ["Destined Death", ["rune of death", "destined death", "death itself", "deathblight"], "The removed Rune of Death — and its theft by the Black Knives."],
    ["The Golden Order", ["golden order", "erdtree", "two fingers", "fundamental"], "Marika's faith: the Erdtree and the Greater Will's order."],
    ["Scarlet Rot", ["scarlet rot", "aeonia", "rotten", "scarlet aeonia"], "The outer god of rot festering in Caelid and within Malenia."],
    ["The Carian Royals", ["carian", "raya lucaria", "glintstone", "full moon"], "Queen Rennala's house and the sorceries of the Academy."],
    ["Frenzied Flame", ["frenzied flame", "yellow flame", "three fingers"], "The chaos that would burn all life and thought to ash."],
    ["A Dynasty of Blood", ["mohg", "formless mother", "blood star"], "Mohg's pursuit of a pure-blood dynasty."],
  ];

  let A, app, nodeById;
  let incoming = new Map();   // target -> [source nodes] via mentions
  let outgoing = new Map();   // source -> [{node,type,w}]
  let mentionsOut = new Map();// source -> [target nodes] via mentions

  function init() {
    app = window.ERA.app; A = app.analysis; nodeById = app.nodeById;
    buildIndexes();
    buildChapterList();
  }

  function buildIndexes() {
    for (const e of app.data.edges) {
      if (e.type === "mentions") {
        if (!incoming.has(e.t)) incoming.set(e.t, []);
        incoming.get(e.t).push(e.s);
        if (!mentionsOut.has(e.s)) mentionsOut.set(e.s, []);
        mentionsOut.get(e.s).push(e.t);
      }
      // structural bonds are recorded from the character's perspective
      pushOut(e.s, e.t, e.type, e.w);
      if (e.type === "shares_location") pushOut(e.t, e.s, e.type, e.w);
    }
  }
  function pushOut(s, t, type, w) {
    if (type === "mentions") return;
    if (!outgoing.has(s)) outgoing.set(s, []);
    outgoing.get(s).push({ id: t, type, w });
  }

  // Data-driven anchor selection: rank by "echoes" — how many item/location
  // descriptions actually name the entity. (Co-location edges are excluded, as
  // they inflate generic roaming enemies rather than storied figures.)
  function buildChapterList() {
    const score = new Map();
    for (const [tid, srcs] of incoming) score.set(tid, srcs.length);

    const pick = (types, n) => {
      const seen = new Set(), out = [];
      const ranked = [...score.entries()]
        .map(([id, sc]) => ({ node: nodeById.get(id), sc }))
        .filter((x) => x.node && types.includes(x.node.type))
        .sort((a, b) => b.sc - a.sc);
      for (const { node } of ranked) {
        const key = node.name.split(",")[0].trim().toLowerCase();
        if (seen.has(key)) continue;
        seen.add(key); out.push(node);
        if (out.length >= n) break;
      }
      return out;
    };

    const groups = [
      ["Demigods & Lords", pick(["boss"], 12)],
      ["Figures of the Lands", pick(["npc"], 8)],
      ["Storied Places", pick(["location"], 6)],
    ];

    const el = document.getElementById("chapter-list");
    el.innerHTML = "";
    for (const [title, items] of groups) {
      if (!items.length) continue;
      const h = document.createElement("div"); h.className = "chapter-group";
      h.innerHTML = `<div class="chapter-grouphead">${title}</div>`;
      for (const node of items) {
        const b = document.createElement("button");
        b.className = "chapter-link";
        const ec = (incoming.get(node.id) || []).length;
        b.innerHTML = `<span>${node.name}</span><span class="chapter-echo">${ec} echoes</span>`;
        b.onclick = () => { selectChapter(node.id); markActive(b); };
        h.appendChild(b);
      }
      el.appendChild(h);
    }
  }
  function markActive(btn) {
    document.querySelectorAll(".chapter-link").forEach((x) => x.classList.remove("active"));
    btn.classList.add("active");
  }

  function tone(s) {
    if (s <= -0.35) return ["Tragic", "#d9484f"];
    if (s < 0.1) return ["Somber", "#b98b54"];
    if (s < 0.4) return ["Resolute", "#c9b06a"];
    return ["Exalted", "#3fb568"];
  }

  function selectChapter(id) {
    const n = nodeById.get(id);
    const echoes = (incoming.get(id) || []).map((s) => nodeById.get(s)).filter(Boolean);
    const speaks = (mentionsOut.get(id) || []).map((t) => nodeById.get(t)).filter(Boolean);
    const bonds = outgoing.get(id) || [];

    // tone over the assembled corpus
    const corpusNodes = [n, ...echoes];
    const sents = corpusNodes.map((x) => x.sent || 0);
    const meanSent = sents.reduce((a, b) => a + b, 0) / (sents.length || 1);
    const [toneWord, toneColor] = tone(meanSent);

    // events
    const blob = corpusNodes.map((x) => (x.lore || "")).join(" ").toLowerCase();
    const foundEvents = EVENTS.filter(([, kws]) => kws.some((k) => blob.includes(k)));

    // bonds grouped
    const bondGroups = {};
    for (const b of bonds) {
      const tn = nodeById.get(b.id); if (!tn) continue;
      (bondGroups[b.type] = bondGroups[b.type] || []).push(tn);
    }

    const echoSorted = echoes.sort((a, b) => (b.lore || "").length - (a.lore || "").length).slice(0, 14);

    const factionBadge = n.faction && n.faction !== "Neither"
      ? `<span class="badge" style="background:${A.FACTION_COLORS[n.faction]}22;color:${A.FACTION_COLORS[n.faction]}">${n.faction}</span>` : "";

    const main = document.getElementById("story-main");
    main.scrollTop = 0;
    main.innerHTML =
      `<div class="dossier">
        <div class="dossier-head">
          <div class="ornament">✦</div>
          <h2>${n.name}</h2>
          <div class="tags">
            <span class="badge" style="background:${A.TYPE_COLORS[n.type]}22;color:${A.TYPE_COLORS[n.type]}">${app.typeLabels[n.type] || n.type}</span>
            ${factionBadge}
            <span class="badge" style="background:${toneColor}22;color:${toneColor}">tone · ${toneWord}</span>
          </div>
        </div>

        ${n.lore ? `<section class="dossier-sec"><h3>The world's account</h3>
          <p class="lore-prime">${n.lore}</p></section>` : ""}

        ${foundEvents.length ? `<section class="dossier-sec"><h3>Events woven through this tale</h3>
          <div class="event-list">${foundEvents.map(([name, , gloss]) =>
            `<div class="event"><b>${name}</b><span>${gloss}</span></div>`).join("")}</div></section>` : ""}

        <section class="dossier-sec"><h3>Echoes in the world <span class="count">${echoes.length}</span></h3>
          <p class="card-note">What items, places and foes say of ${n.name.split(",")[0]} — the story told through their descriptions.</p>
          ${echoSorted.length ? echoSorted.map((e) =>
            `<div class="echo"><div class="echo-head"><span class="dot" style="background:${A.TYPE_COLORS[e.type]}"></span>
               <b data-id="${e.id}" class="echo-name">${e.name}</b>
               <span class="muted">${app.typeLabels[e.type] || e.type}</span></div>
             <p class="echo-lore">${e.lore || "<span class='muted'>(no surviving text)</span>"}</p></div>`).join("")
            : `<div class="muted">No item or place names this entity directly.</div>`}
        </section>

        ${renderBonds(bondGroups)}

        ${speaks.length ? `<section class="dossier-sec"><h3>Speaks of</h3>
          <div class="chips">${speaks.slice(0, 16).map((t) =>
            `<span class="chip link" data-id="${t.id}">${t.name}</span>`).join("")}</div></section>` : ""}

        <button class="reset go-network" data-id="${n.id}">Open “${n.name.split(",")[0]}” in the network →</button>
      </div>`;

    main.querySelectorAll("[data-id]").forEach((el) => {
      if (el.classList.contains("go-network")) {
        el.addEventListener("click", () => app.focusInNetwork(el.dataset.id));
      } else if (el.classList.contains("echo-name") || el.classList.contains("link")) {
        el.style.cursor = "pointer";
        el.addEventListener("click", () => {
          const tid = el.dataset.id;
          if (nodeById.get(tid)) { selectChapter(tid); }
        });
      }
    });
  }

  function renderBonds(groups) {
    const labels = app.edgeLabels;
    const order = ["drops", "located_in", "shares_location"];
    const parts = order.filter((k) => groups[k] && groups[k].length).map((k) => {
      const list = groups[k].slice(0, 14);
      return `<div class="bond-group"><div class="bond-type">${labels[k] || k} <span class="count">${groups[k].length}</span></div>
        <div class="chips">${list.map((t) => `<span class="chip link" data-id="${t.id}">${t.name}</span>`).join("")}</div></div>`;
    });
    if (!parts.length) return "";
    return `<section class="dossier-sec"><h3>Bonds &amp; ties</h3>${parts.join("")}</section>`;
  }

  window.ERA = window.ERA || {};
  window.ERA.story = { init };
})();
