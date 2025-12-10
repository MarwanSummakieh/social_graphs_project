"""
Scrape Elden Ring Wiki (Fextralife) for comprehensive data.

Usage
-----
python scripts/scrape_wiki.py --categories bosses npcs --delay 0.5

This script crawls the Fextralife wiki to gather detailed information including
lore, descriptions, and stats that might be missing from the standard API.
"""
import argparse
import json
import sys
import time
import re
from pathlib import Path
from typing import List, Dict, Set
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://eldenring.wiki.fextralife.com"

# Mapping of logical names to Wiki URLs
CATEGORY_URLS = {
    "bosses": "/Bosses",
    "npcs": "/NPCs",
    "weapons": "/Weapons",
    "armor": "/Armor",
    "talismans": "/Talismans",
    "locations": "/Locations",
    "creatures": "/Creatures+and+Enemies",
    "ashes": "/Ash+of+War",
    "spirits": "/Spirit+Ashes",
    "items": "/Items",
    "lore": "/Lore"
}

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "scraped"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def clean_text(text: str) -> str:
    """Clean whitespace and unwanted characters from text."""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text).strip()

def get_soup(url: str, session: requests.Session) -> BeautifulSoup:
    """Fetch a URL and return a BeautifulSoup object."""
    try:
        response = session.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}", file=sys.stderr)
        return None

def extract_links_from_category(category_path: str, session: requests.Session) -> List[str]:
    """Extract all relevant wiki page links from a category page."""
    url = urljoin(BASE_URL, category_path)
    print(f"Scanning category: {url}")
    soup = get_soup(url, session)
    if not soup:
        return []

    links = set()
    
    # Fextralife usually puts lists in tables or specific divs
    # We look for links in the main content area
    content_div = soup.find('div', {'id': 'wiki-content-block'})
    if not content_div:
        print("Could not find content block.")
        return []

    for a_tag in content_div.find_all('a', href=True):
        href = a_tag['href']
        # Filter out irrelevant links (anchors, external, special pages)
        if href.startswith('/') and not href.startswith('/file/') and ':' not in href:
            # Avoid the category page itself or simple variations
            if href.lower() != category_path.lower():
                links.add(href)
    
    return sorted(list(links))

def scrape_page(page_path: str, session: requests.Session) -> Dict:
    """Scrape content from a single wiki page."""
    url = urljoin(BASE_URL, page_path)
    soup = get_soup(url, session)
    if not soup:
        return None

    data = {
        "url": url,
        "name": "",
        "image": None,
        "description": "",
        "content": [],
        "infobox": {}
    }

    # Extract Name (usually the page title or h1)
    # Fextralife titles often have " | Elden Ring Wiki"
    title_tag = soup.find('title')
    if title_tag:
        data['name'] = title_tag.get_text().split('|')[0].strip()

    content_div = soup.find('div', {'id': 'wiki-content-block'})
    if not content_div:
        return data

    # Extract Image (first significant image in content)
    img = content_div.find('img')
    if img and img.get('src'):
        data['image'] = urljoin(BASE_URL, img['src'])

    # Extract Infobox data (table with class 'wiki_table')
    infobox = content_div.find('table', {'class': 'wiki_table'})
    if infobox:
        for row in infobox.find_all('tr'):
            cols = row.find_all(['th', 'td'])
            if len(cols) >= 2:
                key = clean_text(cols[0].get_text())
                val = clean_text(cols[1].get_text())
                if key and val:
                    data['infobox'][key] = val

    # Extract text content (paragraphs)
    # We skip the infobox and other tables for the main text
    for elem in content_div.find_all(['p', 'h2', 'h3', 'ul']):
        # Skip if inside a table
        if elem.find_parent('table'):
            continue
        
        text = clean_text(elem.get_text())
        if text:
            data['content'].append(text)
    
    # Join first few paragraphs as description
    if data['content']:
        data['description'] = data['content'][0]

    return data

def main():
    parser = argparse.ArgumentParser(description="Scrape Elden Ring Wiki")
    parser.add_argument("--categories", nargs="+", default=list(CATEGORY_URLS.keys()),
                        help=f"Categories to scrape. Options: {', '.join(CATEGORY_URLS.keys())}")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between requests in seconds")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of pages per category (for testing)")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    for category in args.categories:
        if category not in CATEGORY_URLS:
            print(f"Unknown category: {category}. Skipping.")
            continue

        print(f"\n--- Processing Category: {category} ---")
        page_links = extract_links_from_category(CATEGORY_URLS[category], session)
        print(f"Found {len(page_links)} pages in {category}.")

        results = []
        count = 0
        
        for link in page_links:
            if args.limit and count >= args.limit:
                break
            
            print(f"Scraping [{count+1}/{len(page_links)}]: {link}")
            page_data = scrape_page(link, session)
            if page_data:
                page_data['category'] = category
                results.append(page_data)
            
            count += 1
            time.sleep(args.delay)

        # Save results
        output_file = CACHE_DIR / f"{category}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(results)} items to {output_file}")

if __name__ == "__main__":
    main()
