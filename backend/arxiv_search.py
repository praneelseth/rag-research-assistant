# backend/arxiv_search.py

from typing import List, Dict
import urllib.parse
import requests
import xml.etree.ElementTree as ET

def _text(elem, tag):
    """Helper to get the text of a child tag or ''."""
    child = elem.find(tag)
    return child.text.strip() if child is not None and child.text else ""

def search_arxiv(query: str, max_results: int = 10) -> List[Dict]:
    """
    Search arXiv for papers matching the query using Atom feed.
    Returns a list of dicts: title, authors, abstract, pdf_url, published.
    """
    results = []
    try:
        base_url = "http://export.arxiv.org/api/query"
        q = urllib.parse.quote_plus(query)
        url = f"{base_url}?search_query=all:{q}&start=0&max_results={max_results}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}

        for entry in root.findall('atom:entry', ns):
            title = _text(entry, 'atom:title')
            abstract = _text(entry, 'atom:summary')
            published = _text(entry, 'atom:published')
            # Authors
            authors = []
            for author in entry.findall('atom:author', ns):
                name = _text(author, 'atom:name')
                if name:
                    authors.append(name)
            # PDF link
            pdf_url = ""
            for link in entry.findall('atom:link', ns):
                if link.attrib.get('type') == "application/pdf":
                    pdf_url = link.attrib.get('href', '')
                    break
            results.append({
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "pdf_url": pdf_url,
                "published": published.split("T")[0] if published else ""
            })
    except Exception as e:
        print(f"arXiv search error: {e}")
        return []
    return results