"""Curated *Shadow of the Erdtree* dataset.

The public fan API carries no DLC content, so this is a hand-authored,
lore-accurate graph of the DLC's principal figures, their **motives**, and the
relationships that bind them. It is intentionally small and motive-focused: the
"Realm of Shadow" view exists to explain *why* these characters act, and how the
DLC reframes the base game (Marika's origin, Miquella's plan, Mohg's vessel).

Schema mirrors the main dataset (nodes/edges + x/y), with two extras:
``group`` (for coloring) and ``motive`` (the narrative payload). Edges carry a
human-readable ``rel`` label.
"""
from __future__ import annotations

import networkx as nx

# group -> the "side" each figure belongs to (drives color in the UI)
GROUPS = [
    "Miquella's Circle",
    "Marika's Legacy",
    "The Hornsent",
    "Fingers & Outer Gods",
    "Dragons",
    "Frenzy & Abyss",
]

# (id, name, type, group, motive, lore)
_NODES = [
    ("miquella", "Miquella the Kind", "deity", "Miquella's Circle",
     "Shed his flesh, his love, his fear and his doubt to be reborn a benevolent god and raise an Age of Compassion — yet his 'kindness' charms all who behold him into helpless devotion.",
     "The empyrean twin of Malenia. Climbing the great tower of Enir-Ilim, Miquella abandons the Golden Order and casts away pieces of himself to ascend."),
    ("radahn", "Promised Consort Radahn", "boss", "Miquella's Circle",
     "Bound by an oath sworn in childhood to become Miquella's consort and Elden Lord — the martial throne upon which the new god's order would stand.",
     "Starscourge Radahn, returned as Miquella's promised consort, his soul housed in the body of Mohg."),
    ("st_trina", "St. Trina", "npc", "Miquella's Circle",
     "Miquella's discarded love, given her own will. Alone among his works she would STOP him — offering an eternal sleep to end his ascent before its cost is paid.",
     "The cast-off other self of Miquella; a voice of sleep and of mercy turned against its maker."),
    ("thiollier", "Thiollier", "npc", "Miquella's Circle",
     "A devotee of St. Trina who seeks her doctrine of sleep and death, doubting the bright promise of Miquella.",
     "A pale aristocrat-turned-disciple who follows Trina's whispered path."),
    ("leda", "Leda, Needle Knight", "npc", "Miquella's Circle",
     "Self-appointed enforcer of Miquella's faithful — she will purge any companion she deems a threat to her lord's design.",
     "A knight who gathers Miquella's followers, then turns her blade on them in his name."),
    ("malenia", "Malenia, Blade of Miquella", "boss", "Miquella's Circle",
     "Sworn shield to her twin Miquella; her undefeated blade and her bloom of scarlet rot are the price of that devotion.",
     "Miquella's twin, the Severed; her rot scoured Caelid in her duel with Radahn."),
    ("romina", "Romina, Saint of the Bud", "boss", "Miquella's Circle",
     "Guardian of the rot-choked path to Miquella's tower; a saint who embraced scarlet rot as devotion.",
     "Saint of the Bud, barring the gate of divine beasts on the road to Enir-Ilim."),
    ("enir_ilim", "Enir-Ilim", "location", "Miquella's Circle",
     "The Hornsent's great tower reaching for the heavens — the stage of Miquella's godhood.",
     "The spiraling tower-city where one might 'stand before the gate' and become divine."),

    ("mohg", "Mohg, Lord of Blood", "boss", "Frenzy & Abyss",
     "Abducted Miquella's cocoon to raise a Mohgwyn dynasty of blood — unwittingly preparing the vessel Miquella needed.",
     "Lord of Blood, whose corpse becomes the body of the Promised Consort."),
    ("ansbach", "Sir Ansbach", "npc", "Frenzy & Abyss",
     "A loyal servant of Mohg's pure-blood dynasty who follows Miquella's trail to weigh — and perhaps avenge — his fallen lord.",
     "Pureblood Knight of the Mohgwyn dynasty, courteous even in ruin."),
    ("midra", "Midra, Lord of Frenzied Flame", "boss", "Frenzy & Abyss",
     "A gentle scholar driven to despair until the Frenzied Flame took root — the yearning to burn all life and thought to ash.",
     "Host of the Frenzied Flame within the Manus Celes library of the Shadow Keep."),

    ("messmer", "Messmer the Impaler", "boss", "Marika's Legacy",
     "Marika's abandoned son, gifted a sealed abyssal serpent-flame and commanded to lead a crusade that burned the Hornsent — a war his mother erased from memory.",
     "The Impaler, his flame sealed behind a divine seal, ruler of the Shadow Keep."),
    ("rellana", "Rellana, Twin Moon Knight", "boss", "Marika's Legacy",
     "Messmer's elder sister who left for Liurnia and mastered Carian moon sorcery, swearing her swords to a love beyond the crusade.",
     "Twin Moon Knight of Castle Ensis, wielder of dark and light moon blades."),
    ("gaius", "Commander Gaius", "boss", "Marika's Legacy",
     "A boar-riding commander of Messmer's host, raised among the war-orphans, sworn to the crusade.",
     "Commander of Messmer's army, guarding the approach to the Shadow Keep."),
    ("dancing_lion", "Divine Beast Dancing Lion", "boss", "Marika's Legacy",
     "A storm-wreathed divine beast bound in service, guarding the rites of Belurat.",
     "A two-souled divine beast of wind, frost and lightning."),
    ("marika", "Queen Marika", "deity", "Marika's Legacy",
     "A Hornsent shaman who, after her people's slaughter, ascended to godhood and took terrible vengeance — sealing the shadow realm and its sins out of history.",
     "The Eternal Queen; her ascent and her crusade are the wound the DLC reopens."),

    ("hornsent", "The Hornsent", "npc", "The Hornsent",
     "The horned people who once ruled the shadow land and made saints by sealing the living in jars; persecuted and erased, they cry for justice — or for vengeance.",
     "Once the dominant civilization of the Land of Shadow, undone by Marika's crusade."),
    ("scadutree", "The Shadow Tree (Scadutree)", "location", "The Hornsent",
     "The shadow cast by the Erdtree; the source of grace in the Land of Shadow, around which the realm's faith turns.",
     "The Scadutree, whose blessing strengthens those who walk the realm of shadow."),

    ("ymir", "Count Ymir, Mother of Fingers", "npc", "Fingers & Outer Gods",
     "A sage who would murder the messengers of the Greater Will and birth a NEW god and new Fingers of his own making.",
     "Master Sage of the Church of the Bud, scheming to supplant the outer god's guidance."),
    ("metyr", "Metyr, Mother of Fingers", "boss", "Fingers & Outer Gods",
     "The Greater Will's severed messenger and mother of the Two/Three Fingers — abandoned by her god, she signals into a silent void.",
     "An ancient star-born vessel, origin of the Fingers that relayed the Greater Will."),
    ("greater_will", "The Greater Will", "deity", "Fingers & Outer Gods",
     "The distant outer god whose order shaped the Lands Between — and who long ago fell silent, abandoning its own messenger.",
     "The outer god behind the Golden Order; its absence drives Ymir and Metyr's grief."),

    ("bayle", "Bayle the Dread", "boss", "Dragons",
     "An ancient dragon who turned on his own kind out of dread of the Greater Will — fear made into a tyrant of storms and rot.",
     "The Dread, scarred foe of Placidusax's line, hunted through the Jagged Peak."),
    ("florissax", "Dragon Priestess Florissax", "npc", "Dragons",
     "Last loyal drake-priestess who would restore the dragons' glory and see Bayle's dread undone.",
     "Florissax, keeper of dragon communion and ally against Bayle."),
]

# (source, target, relationship label)
_EDGES = [
    ("miquella", "radahn", "promised consort"),
    ("miquella", "st_trina", "discarded love"),
    ("miquella", "malenia", "twin"),
    ("miquella", "enir_ilim", "ascends at"),
    ("miquella", "leda", "is served by"),
    ("radahn", "mohg", "soul housed in body of"),
    ("mohg", "miquella", "abducted (cocoon)"),
    ("ansbach", "mohg", "loyal to"),
    ("st_trina", "miquella", "seeks to stop"),
    ("thiollier", "st_trina", "follows"),
    ("malenia", "romina", "rot kin"),
    ("romina", "enir_ilim", "guards path to"),
    ("messmer", "marika", "abandoned child of"),
    ("messmer", "hornsent", "crusade against"),
    ("messmer", "rellana", "sibling"),
    ("messmer", "gaius", "commands"),
    ("messmer", "dancing_lion", "commands"),
    ("marika", "hornsent", "born of"),
    ("marika", "scadutree", "sealed the realm of"),
    ("hornsent", "enir_ilim", "built"),
    ("hornsent", "scadutree", "worship"),
    ("ymir", "metyr", "child / would supplant"),
    ("metyr", "greater_will", "messenger of"),
    ("greater_will", "metyr", "abandoned"),
    ("greater_will", "marika", "raised as vessel"),
    ("bayle", "florissax", "dread enemy of"),
    ("rellana", "radahn", "sworn love (lore)"),
]


def build_dlc() -> dict:
    g = nx.Graph()
    for nid, *_ in _NODES:
        g.add_node(nid)
    for s, t, _ in _EDGES:
        g.add_edge(s, t)
    pos = nx.spring_layout(g, k=0.9, iterations=200, seed=7)

    nodes = []
    for nid, name, ntype, group, motive, lore in _NODES:
        x, y = pos[nid]
        nodes.append({
            "id": nid, "name": name, "type": ntype, "group": group,
            "motive": motive, "lore": lore,
            "x": round(float(x) * 100, 2), "y": round(float(y) * 100, 2),
        })
    edges = [{"s": s, "t": t, "rel": rel} for s, t, rel in _EDGES]

    from collections import Counter
    return {
        "meta": {
            "title": "Realm of Shadow — Shadow of the Erdtree",
            "node_count": len(nodes),
            "edge_count": len(edges),
            "groups": GROUPS,
            "group_counts": dict(Counter(n["group"] for n in nodes)),
            "note": "Curated, lore-accurate dataset (not from the API).",
        },
        "nodes": nodes,
        "edges": edges,
    }
