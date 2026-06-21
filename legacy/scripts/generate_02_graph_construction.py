
import json
import os

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 02. Graph Construction\n",
    "\n",
    "## Overview\n",
    "This notebook processes the raw JSON data collected in step 01 and constructs a NetworkX graph.\n",
    "It performs the following steps:\n",
    "1.  **Node Creation**: Converts items, NPCs, locations, etc., into graph nodes.\n",
    "2.  **Edge Creation**: Links nodes based on mentions in descriptions AND scraped wiki references.\n",
    "3.  **Attribute Extraction**: Extracts key attributes like 'scaling_type' (Int vs. Faith) and 'fate_type' (Bell Bearings).\n",
    "4.  **Export**: Saves the processed graph as `nodes.csv` and `edges.csv` for analysis.\n",
    "\n",
    "## Setup"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import networkx as nx\n",
    "import json\n",
    "import re\n",
    "from pathlib import Path\n",
    "from typing import List, Dict, Any\n",
    "from bs4 import BeautifulSoup\n",
    "\n",
    "# Paths\n",
    "RAW_DATA_DIR = Path(\"../data/raw\")\n",
    "WIKI_DATA_DIR = RAW_DATA_DIR / \"wiki_html\"\n",
    "PROCESSED_DATA_DIR = Path(\"../data/processed\")\n",
    "PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "# Node Types Mapping\n",
    "NODE_CONFIG = {\n",
    "    \"items\": \"item\",\n",
    "    \"weapons\": \"weapon\",\n",
    "    \"npcs\": \"npc\",\n",
    "    \"locations\": \"location\",\n",
    "    \"bosses\": \"boss\",\n",
    "    \"armors\": \"armor\",\n",
    "    \"talismans\": \"talisman\",\n",
    "    \"incantations\": \"spell\",\n",
    "    \"creatures\": \"creature\",\n",
    "    \"shields\": \"shield\"\n",
    "}\n",
    "\n",
    "print(\"Setup complete.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Load and Process Nodes\n",
    "We load each JSON file and convert it into a standardized DataFrame of nodes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_nodes() -> pd.DataFrame:\n",
    "    all_nodes = []\n",
    "    \n",
    "    for filename, node_type in NODE_CONFIG.items():\n",
    "        filepath = RAW_DATA_DIR / f\"{filename}.json\"\n",
    "        if not filepath.exists():\n",
    "            print(f\"Skipping {filename} (not found)\")\n",
    "            continue\n",
    "            \n",
    "        with open(filepath, 'r', encoding='utf-8') as f:\n",
    "            data = json.load(f)\n",
    "            \n",
    "        print(f\"Processing {len(data)} {node_type}s...\")\n",
    "        \n",
    "        for item in data:\n",
    "            # Create a unique ID\n",
    "            node_id = item.get(\"id\") or f\"{node_type}:{item.get('name')}\"\n",
    "            \n",
    "            # Extract description (handle different field names)\n",
    "            description = item.get(\"description\") or item.get(\"effect\") or \"\"\n",
    "            if isinstance(description, list):\n",
    "                description = \" \".join(description)\n",
    "                \n",
    "            # Extract extra attributes (scaling, etc.)\n",
    "            extra = {k: v for k, v in item.items() if k not in [\"id\", \"name\", \"description\", \"effect\", \"image\"]}\n",
    "            \n",
    "            all_nodes.append({\n",
    "                \"node_id\": node_id,\n",
    "                \"name\": item.get(\"name\", \"Unknown\"),\n",
    "                \"type\": node_type,\n",
    "                \"description\": description,\n",
    "                \"image\": item.get(\"image\"),\n",
    "                \"extra\": json.dumps(extra) # Store as JSON string for CSV\n",
    "            })\n",
    "            \n",
    "    return pd.DataFrame(all_nodes)\n",
    "\n",
    "nodes_df = load_nodes()\n",
    "print(f\"Total nodes: {len(nodes_df)}\")\n",
    "nodes_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Feature Extraction\n",
    "Now we extract specific features needed for our analysis:\n",
    "- **Scaling Type**: Does a weapon scale with Int, Faith, or both?\n",
    "- **Fate Type**: Is this item a Bell Bearing, Remembrance, or imply death?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def extract_scaling(row):\n",
    "    if row['type'] not in ['weapon', 'shield', 'spell']:\n",
    "        return None\n",
    "        \n",
    "    try:\n",
    "        extra = json.loads(row['extra'])\n",
    "        scales = extra.get('scalesWith', [])\n",
    "        if not scales:\n",
    "            return None\n",
    "            \n",
    "        scale_names = {s['name'].strip().lower() for s in scales}\n",
    "        \n",
    "        has_int = 'int' in scale_names\n",
    "        has_fai = 'fai' in scale_names or 'fth' in scale_names\n",
    "        \n",
    "        if has_int and has_fai:\n",
    "            return 'Int/Fth'\n",
    "        elif has_int:\n",
    "            return 'Intelligence'\n",
    "        elif has_fai:\n",
    "            return 'Faith'\n",
    "        else:\n",
    "            return 'Other'\n",
    "    except:\n",
    "        return None\n",
    "\n",
    "nodes_df['scaling_type'] = nodes_df.apply(extract_scaling, axis=1)\n",
    "print(nodes_df['scaling_type'].value_counts())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def extract_fate(row):\n",
    "    name = str(row['name']).lower()\n",
    "    desc = str(row['description']).lower()\n",
    "    \n",
    "    if 'bell bearing' in name or 'remembrance' in name:\n",
    "        return 'explicit_fate_item'\n",
    "    \n",
    "    death_keywords = [\"died\", \"slain\", \"killed\", \"corpse\", \"remains\", \"left behind\", \"last words\", \"perished\"]\n",
    "    if any(k in desc for k in death_keywords):\n",
    "        return 'implied_death'\n",
    "        \n",
    "    return None\n",
    "\n",
    "nodes_df['fate_type'] = nodes_df.apply(extract_fate, axis=1)\n",
    "print(nodes_df['fate_type'].value_counts())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Edge Creation (Mentions + Wiki)\n",
    "We create edges from two sources:\n",
    "1.  **Description Mentions**: Text analysis of item descriptions.\n",
    "2.  **Wiki References**: Hyperlinks extracted from scraped wiki pages."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def create_edges(nodes_df):\n",
    "    edges = []\n",
    "    \n",
    "    # --- 1. Description Mentions ---\n",
    "    target_types = ['npc', 'location', 'boss', 'creature']\n",
    "    targets = nodes_df[nodes_df['type'].isin(target_types)]\n",
    "    \n",
    "    target_map = {}\n",
    "    for _, row in targets.iterrows():\n",
    "        name = row['name']\n",
    "        node_id = row['node_id']\n",
    "        target_map[name] = node_id\n",
    "        if \",\" in name:\n",
    "            simple_name = name.split(\",\")[0].strip()\n",
    "            if len(simple_name) > 3:\n",
    "                target_map[simple_name] = node_id\n",
    "            \n",
    "    sorted_names = sorted(target_map.keys(), key=len, reverse=True)\n",
    "    pattern = re.compile(r'\\b(' + '|'.join(map(re.escape, sorted_names)) + r')(?:\\'s)?\\b')\n",
    "    \n",
    "    print(\"Scanning descriptions for mentions...\")\n",
    "    for _, row in nodes_df.iterrows():\n",
    "        source_id = row['node_id']\n",
    "        desc = str(row['description'])\n",
    "        name = str(row['name'])\n",
    "        \n",
    "        if not desc:\n",
    "            continue\n",
    "            \n",
    "        found_names = set(pattern.findall(desc))\n",
    "        found_in_name = set(pattern.findall(name))\n",
    "        found_names.update(found_in_name)\n",
    "        \n",
    "        for name_match in found_names:\n",
    "            if isinstance(name_match, tuple): name_match = name_match[0]\n",
    "            target_id = target_map.get(name_match)\n",
    "            if target_id and source_id != target_id:\n",
    "                edges.append({'source': source_id, 'target': target_id, 'type': 'mentioned_in', 'weight': 1})\n",
    "\n",
    "    # --- 2. Wiki References ---\n",
    "    print(\"Scanning scraped wiki pages for references...\")\n",
    "    # Map all names to IDs for wiki linking\n",
    "    name_to_id = {row['name'].lower(): row['node_id'] for _, row in nodes_df.iterrows()}\n",
    "    \n",
    "    if WIKI_DATA_DIR.exists():\n",
    "        for html_file in WIKI_DATA_DIR.glob(\"*.html\"):\n",
    "            source_name = html_file.stem.replace(\"_\", \" \")\n",
    "            source_id = name_to_id.get(source_name.lower())\n",
    "            \n",
    "            if not source_id:\n",
    "                continue\n",
    "                \n",
    "            with open(html_file, 'r', encoding='utf-8') as f:\n",
    "                soup = BeautifulSoup(f, 'html.parser')\n",
    "                \n",
    "            # Find all links\n",
    "            for link in soup.find_all('a'):\n",
    "                href = link.get('href', '')\n",
    "                text = link.get_text().strip()\n",
    "                \n",
    "                # Check if link text matches a known node\n",
    "                target_id = name_to_id.get(text.lower())\n",
    "                if target_id and target_id != source_id:\n",
    "                    edges.append({'source': source_id, 'target': target_id, 'type': 'wiki_reference', 'weight': 2})\n",
    "                    \n",
    "    return pd.DataFrame(edges)\n",
    "\n",
    "edges_df = create_edges(nodes_df)\n",
    "print(f\"Created {len(edges_df)} edges.\")\n",
    "edges_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Export\n",
    "Save the nodes and edges to CSV."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "nodes_df.to_csv(PROCESSED_DATA_DIR / \"nodes.csv\", index=False)\n",
    "edges_df.to_csv(PROCESSED_DATA_DIR / \"edges.csv\", index=False)\n",
    "print(\"Data exported successfully.\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('c:/social_graphs_project/notebooks/02_graph_construction.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook_content, f, indent=1)
