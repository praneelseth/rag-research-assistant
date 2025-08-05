# backend/arxiv_search.py

from typing import List, Dict
import urllib.parse
import requests
import feedparser

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
        feed = feedparser.parse(resp.text)
        for entry in feed.entries:
            title = entry.title.strip()
            authors = [a.name for a in entry.authors] if hasattr(entry, "authors") else []
            abstract = entry.summary.strip() if hasattr(entry, "summary") else ""
            # Find the PDF link
            pdf_url = ""
            for link in entry.links:
                if link.type == "application/pdf":
                    pdf_url = link.href
                    break
            published = entry.published.split("T")[0] if hasattr(entry, "published") else ""
            results.append({
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "pdf_url": pdf_url,
                "published": published
            })
    except Exception as e:
        print(f"arXiv search error: {e}")
        return []
    return results