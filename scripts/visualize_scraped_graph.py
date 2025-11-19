"""
Build and visualize a network graph from scraped Elden Ring data.

Usage
-----
python scripts/visualize_scraped_graph.py --categories bosses npcs --output graph.png

This script loads scraped JSON data, builds a network where edges represent
mentions in descriptions, and saves a visualization.
"""
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx

SCRAPED_DIR = Path(__file__).resolve().parents[1] / "data" / "scraped"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"

def load_data(categories: List[str]) -> List[dict]:
    """Load scraped data for specified categories."""
    data = []
    for category in categories:
        path = SCRAPED_DIR / f"{category}.json"
        if not path.exists():
            print(f"Warning: {path} not found. Skipping.")
            continue
        
        try:
            items = json.loads(path.read_text(encoding="utf-8"))
            # Add category tag to each item
            for item in items:
                item['category'] = category
            data.extend(items)
        except json.JSONDecodeError:
            print(f"Error decoding {path}.")
    return data

def normalize_name(name: str) -> str:
    """Normalize name for matching (lowercase, strip)."""
    return name.lower().strip()

def build_graph(data: List[dict]) -> nx.DiGraph:
    """Build a directed graph from data based on description mentions."""
    G = nx.DiGraph()
    
    # 1. Add all nodes
    # Create a lookup map for name -> item
    name_map = {}
    
    for item in data:
        name = item.get("name")
        if not name:
            continue
        
        # Use name as ID for simplicity in this visualization
        G.add_node(name, category=item.get("category"), url=item.get("url"))
        name_map[normalize_name(name)] = name

    print(f"Added {G.number_of_nodes()} nodes.")

    # 2. Create edges based on mentions
    # This is O(N^2) roughly, but N is small enough for this dataset (~1-2k items)
    edge_count = 0
    
    for item in data:
        source_name = item.get("name")
        if not source_name:
            continue
            
        description = item.get("description", "")
        if not description:
            continue
            
        desc_lower = description.lower()
        
        # Check for mentions of other items
        for target_norm, target_real_name in name_map.items():
            if source_name == target_real_name:
                continue
            
            # Simple substring match - can be improved with regex boundaries
            # We use a simple check to avoid matching short common words if any exist
            if len(target_norm) > 3 and target_norm in desc_lower:
                G.add_edge(source_name, target_real_name)
                edge_count += 1

    print(f"Added {edge_count} edges.")
    return G

def visualize_graph(G: nx.DiGraph, output_path: Path):
    """Visualize the graph and save to file."""
    plt.figure(figsize=(12, 12))
    
    # Compute layout
    pos = nx.spring_layout(G, k=0.15, iterations=20)
    
    # Color nodes by category
    categories = set(nx.get_node_attributes(G, 'category').values())
    color_map = plt.cm.get_cmap('tab10', len(categories))
    category_to_color = {cat: color_map(i) for i, cat in enumerate(categories)}
    
    node_colors = [category_to_color.get(G.nodes[n].get('category'), 'gray') for n in G.nodes()]
    
    # Draw
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.2, arrows=True)
    
    # Only label high-degree nodes to avoid clutter
    degrees = dict(G.degree())
    top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:20]
    labels = {n: n for n in top_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='black')
    
    plt.title("Elden Ring Entity Network (Description Mentions)")
    plt.axis('off')
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Graph saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize Scraped Graph")
    parser.add_argument("--categories", nargs="+", default=["bosses", "npcs"], 
                        help="Categories to include in the graph")
    parser.add_argument("--output", type=str, default="scraped_graph.png",
                        help="Output filename")
    args = parser.parse_args()

    print(f"Loading data for: {args.categories}")
    data = load_data(args.categories)
    
    if not data:
        print("No data found. Did you run scrape_wiki.py?")
        return

    G = build_graph(data)
    
    # Remove isolated nodes for cleaner visualization
    G.remove_nodes_from(list(nx.isolates(G)))
    print(f"Nodes after removing isolates: {G.number_of_nodes()}")

    output_path = OUTPUT_DIR / args.output
    visualize_graph(G, output_path)

if __name__ == "__main__":
    main()
