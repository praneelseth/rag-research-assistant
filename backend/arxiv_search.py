# backend/arxiv_search.py

from typing import List, Dict
import arxiv

def search_arxiv(query: str, max_results: int = 10) -> List[Dict]:
    """
    Search arXiv for papers matching the query.
    Returns a list of dicts: title, authors, abstract, pdf_url, published.
    """
    results = []
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        for result in search.results():
            results.append({
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "abstract": result.summary,
                "pdf_url": result.pdf_url,
                "published": str(result.published.date())
            })
    except Exception as e:
        print(f"arXiv search error: {e}")
    return results