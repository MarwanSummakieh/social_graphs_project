# Research Proposal: The Architecture of Fate in Elden Ring

## Research Question
"To what extent do game mechanics (specifically stat scaling and item utility) dictate the narrative topology and character fates of the Elden Ring universe?"

## The "Research Finding"
We argue that gameplay mechanics are not separate from the lore; they are the primary architects of the world's structure.
1.  **The Bipolar World:** The "Intelligence vs. Faith" stat mechanic creates a hard structural and semantic split.
2.  **The Tragedy of Utility:** The "Bell Bearing" mechanic structurally condemns functional characters to tragic fates.
3.  **The Illusion of Choice:** "Key Items" represent rare structural "bridges" that allow the player to collapse the Bipolar World.

---

## Project Assignment A: Video Pitch Outline

**Hook:** "Can a stat number on a sword predict the history of a kingdom?"

**Data:**
-   Elden Ring API (Nodes: Items/NPCs, Edges: Mentions/Locations).

**Method:**
-   Combining Network Science (to see the structure) and NLP (to read the story).

**Preliminary Finding:**
-   Show the "Blue vs. Gold" network visualization (from Notebook 03).
-   Show the list of "Tragic Merchants" (from Notebook 03/04).

---

## Project Assignment B: Paper Structure

**1. Introduction**
-   Game studies often separate "Ludology" (mechanics) and "Narratology" (story).
-   We use data science to bridge them.

**2. Methods**
-   Graph construction from API (Notebook 01 & 02).
-   TF-IDF & Community Detection (Notebook 03 & 04).

**3. Results**
-   **Fig 1:** The Int/Faith Network Split (Visual proof of the Schism).
-   **Fig 2:** Word Clouds for Int vs. Faith (Semantic proof).
-   **Fig 3:** The "Fate Graph" showing merchants clustering around Bell Bearings.

**4. Discussion**
-   The game's structure enforces a deterministic worldview where mechanics (stats/shops) are destiny.

---

## Notebook Workflow

1.  **`01_data_collection.ipynb`**: Fetch raw data.
2.  **`02_graph_construction.ipynb`**: Build the graph and extract "Fate" attributes.
3.  **`03_network_analysis.ipynb`**: Analyze the "Schism" and "Tragic Hubs".
4.  **`04_nlp_analysis.ipynb`**: Analyze the vocabulary of the factions.
