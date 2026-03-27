"""
data_pipeline/fetch_corpus.py
S2 Bulk Search ingestion pipeline.

Usage:
    python data_pipeline/fetch_corpus.py          # incremental (default)
    python data_pipeline/fetch_corpus.py --full   # full refresh
"""
import logging
import os
import sys
import time
import random
import argparse
from datetime import datetime, timedelta
from typing import List, Optional

import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from data_pipeline.schema import PaperRecord, create_db, upsert_paper
except ImportError:
    # Script mode: `python data_pipeline/fetch_corpus.py` — project root not on sys.path.
    # Insert it and retry. This is a no-op when imported as a module.
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from data_pipeline.schema import PaperRecord, create_db, upsert_paper

# Load .env so S2_API_KEY is available even when run as a standalone script
load_dotenv()

logger = logging.getLogger(__name__)

# Semantic Scholar fields to request via bulk search.
# NOTE: authors.citationCount is NOT supported by the bulk endpoint (causes 400).
# Author-level citation counts can be fetched via the per-paper detail endpoint
# in a follow-up enrichment pass (P2 backlog, G-01).
_FIELDS = [
    "paperId",
    "title",
    "abstract",
    "authors",          # returns authorId + name by default
    "year",
    "publicationDate",
    "venue",
    "citationCount",
    "externalIds",
    "openAccessPdf",
    "s2FieldsOfStudy",
]


def parse_s2_paper(raw: dict) -> Optional[PaperRecord]:
    """
    Convert a raw S2 API dict to a PaperRecord.
    Returns None if the paper has no arXiv ID (non-arXiv papers skipped).

    Note: max_author_citations is set to 0 because authors.citationCount is
    not available from the bulk search endpoint. Can be enriched later.
    """
    try:
        ids = raw.get("externalIds") or {}
        arxiv_id = ids.get("ArXiv", "")
        if not arxiv_id:
            return None

        authors = raw.get("authors") or []
        pdf_url = (raw.get("openAccessPdf") or {}).get(
            "url", f"https://arxiv.org/pdf/{arxiv_id}"
        )
        pub_date = (
            raw.get("publicationDate")
            or str(raw.get("year", "2024")) + "-01-01"
        )
        
        # S2 frequently returns 'ArXiv' as the native venue for preprints. Strip it out.
        s2_venue = raw.get("venue")
        if s2_venue and s2_venue.lower() in ("arxiv", "arxiv.org"):
            s2_venue = None

        return PaperRecord(
            arxiv_id=arxiv_id,
            s2_id=raw.get("paperId", ""),
            title=(raw.get("title") or "").replace("\n", " ").strip(),
            abstract=(raw.get("abstract") or "").replace("\n", " ").strip(),
            authors=[a.get("name", "") for a in authors],
            submitted_date=pub_date,
            venue=s2_venue,
            citation_count=raw.get("citationCount", 0) or 0,
            max_author_citations=0,  # not available from bulk endpoint
            pdf_url=pdf_url,
            arxiv_url=f"https://arxiv.org/abs/{arxiv_id}",
            fields_of_study=[
                f.get("category", "")
                for f in (raw.get("s2FieldsOfStudy") or [])
            ],
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("parse error: %s", exc)
        return None


_S2_BULK_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
MAX_PAPERS_TO_FETCH = 200_000

@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(min=2, max=30),
    reraise=True,
)
def _fetch_page(url: str, params: dict, headers: dict) -> dict:
    """Single paginated GET with retry."""
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_papers_bulk(
    api_key: Optional[str] = None,
    max_papers: int = MAX_PAPERS_TO_FETCH,
    **_,
) -> List[PaperRecord]:
    """
    Pull papers from S2 via Bulk Search REST API.
    Returns deduplicated list of PaperRecord (arXiv papers only).
    """
    resolved_key = api_key or os.getenv("S2_API_KEY")
    headers = {"x-api-key": resolved_key} if resolved_key else {}
    # Define a 60-day rolling window for efficiency. (Adds 1-month buffer
    # for indexing delays, while still safely covering Streamlit's "Last Month" limit)
    start_date_str = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    try:
        from app import ARXIV_CODE_TO_NAME, CONFERENCE_KEYWORDS, JOURNAL_KEYWORDS
        subcats = list(ARXIV_CODE_TO_NAME.values())
    except ImportError:
        subcats = [
            "Artificial Intelligence", "Machine Learning", "Human-Computer Interaction",
            "Computation and Language", "Computer Vision and Pattern Recognition",
            "Robotics", "Information Retrieval", "Neural and Evolutionary Computing",
            "Software Engineering", "Cryptography and Security",
            "Data Structures and Algorithms", "Databases",
            "Social and Information Networks", "Multimedia",
            "Information Theory", "Performance", "Multiagent Systems"
        ]

    # "None" designates the base Computer Science sweep, others are explicit keyword sub-categories
    queries = [None] + subcats

    seen: set = set()
    results: List[PaperRecord] = []
    
    for q in queries:
        if len(results) >= max_papers:
            break
            
        params: dict = {
            "publicationDateOrYear": f"{start_date_str}:",
            "fields": ",".join(_FIELDS),
            "limit": 1000,
        }
        
        if q is None:
            params["fieldsOfStudy"] = "Computer Science"
            logger.info("Fetching generic 'Computer Science' base loop...")
        else:
            params["query"] = q
            logger.info("Fetching sub-category explicitly: '%s'", q)
            
        while True:
            data = _fetch_page(_S2_BULK_URL, params, headers)
            batch = data.get("data") or []
            
            for raw in batch:
                paper = parse_s2_paper(raw)
                if paper and paper.arxiv_id not in seen:
                    seen.add(paper.arxiv_id)
                    results.append(paper)
                    if len(results) >= max_papers:
                        break
            
            token = data.get("token")
            logger.info("Batch parsed: %d raw | total arXiv so far: %d | Token: %s", len(batch), len(results), bool(token))
            
            if not token or len(results) >= max_papers:
                break
                
            params["token"] = token
            time.sleep(1.1)

    logger.info("Fetched %d arXiv papers across all sub-categories and base CS", len(results))
    return results


def fetch_fresh_arxiv_papers(days: int = 30) -> List[PaperRecord]:
    """
    Stage 1 'Scout': Pull recent preprints directly from ArXiv so they appear immediately.
    """
    import feedparser
    from urllib.parse import quote
    
    try:
        from app import ARXIV_CODE_TO_NAME, extract_venue
        cats = list(ARXIV_CODE_TO_NAME.keys())
        extract_venue_func = extract_venue
    except ImportError:
        cats = [
            "cs.AI", "cs.LG", "cs.HC", "cs.CL", "cs.CV", "cs.RO", "cs.IR", "cs.NE", "cs.SE",
            "cs.CR", "cs.DS", "cs.DB", "cs.SI", "cs.MM", "cs.IT", "cs.PF", "cs.MA"
        ]
        extract_venue_func = None

    query_str = " OR ".join(f"cat:{c}" for c in cats)
    cutoff_date = (datetime.now() - timedelta(days=days))
    
    logger.info("Starting Stage 1 ArXiv Scout targeting last %d days...", days)
    
    results = []
    seen = set()
    start = 0
    max_results = 1000
    backoff = 3
    
    while True:
        # Enforce ArXiv's absolute minimum 3-second delay
        time.sleep(3.1)
        
        url = f"http://export.arxiv.org/api/query?search_query=({quote(query_str)})&sortBy=submittedDate&sortOrder=descending&start={start}&max_results={max_results}"
        
        # Adaptive Stealth 3 & 4: User-Agent Identity & True Exponential Backoff
        import requests
        try:
            headers = {"User-Agent": "ResearchAgent/2.0 (mailto:admin@example.com)"}
            r = requests.get(url, headers=headers, timeout=60)
            r.raise_for_status()
            feed_text = r.text
            backoff = 3 # Reset on success
        except Exception as http_err:
            logger.error("ArXiv API hit a limit or failed: %s. Backing off for %ds...", http_err, backoff)
            time.sleep(backoff)
            if backoff < 300:
                backoff *= 2
            continue
                
        feed = feedparser.parse(feed_text)
        
        # 1. Avoid error pages being treated as entries
        if feed.get("bozo_exception"):
            logger.error("ArXiv API returned malformed feed (possible error page). Backing off for %ds...", backoff)
            time.sleep(backoff)
            if backoff < 300:
                backoff *= 2
            continue
            
        if not feed.get("entries"):
            break
            
        oldest_date_in_batch = None
        for entry in feed.get("entries", []):
            arxiv_or_full = entry.get("id", "")
            if not arxiv_or_full:
                continue
                
            if "/abs/" in arxiv_or_full:
                arxiv_id = arxiv_or_full.split("/abs/")[-1].split("v")[0]
            else:
                arxiv_id = arxiv_or_full
                
            # Filter out error documents mimicking entries
            if len(arxiv_id) > 30 or " " in arxiv_id or "error" in arxiv_id.lower():
                continue
                
            if arxiv_id in seen:
                continue
            seen.add(arxiv_id)
            
            # ArXiv feed uses 'published' or 'updated'. Use .get() defensively.
            pub_date_str = entry.get("published") or entry.get("updated")
            if not pub_date_str:
                pub_date = datetime.now()
            else:
                try:
                    # ArXiv standard format: 2026-03-21T21:30:11Z
                    pub_date = datetime.strptime(pub_date_str, "%Y-%m-%dT%H:%M:%SZ")
                except Exception:
                    pub_date = datetime.now()
                
            # Track the oldest to know when to terminate pagination
            if oldest_date_in_batch is None or pub_date < oldest_date_in_batch:
                oldest_date_in_batch = pub_date
                
            # If older than cutoff, don't append, but we might keep searching the page
            if pub_date < cutoff_date:
                continue
                
            authors = [a.get("name", "") for a in entry.get("authors", []) if a.get("name")]
            
            # 3. Extract correct arXiv tags (cs.AI, cs.RO, etc.) directly into fields_of_study
            extracted_tags = []
            for t in entry.get("tags", []):
                term = t.get("term")
                if term and not term.startswith("http"):
                    extracted_tags.append(term)
                    
            # 4. Extract venue natively from arXiv comments if present
            comment = entry.get("arxiv_comment", "")
            journal_ref = entry.get("arxiv_journal_ref", "")
            final_venue = None
            if extract_venue_func:
                val = extract_venue_func(comment) or extract_venue_func(journal_ref)
                if val:
                    final_venue = val
            
            results.append(PaperRecord(
                arxiv_id=arxiv_id,
                s2_id="", # Null until S2 Stage 2 populates it
                title=entry.get("title", "No Title").replace('\n', ' ').strip(),
                abstract=entry.get("summary", "No Abstract").replace('\n', ' ').strip(),
                authors=authors,
                # 2. Maintain strict YYYY-MM-DD format dynamically
                submitted_date=pub_date.strftime("%Y-%m-%d"),
                venue=final_venue,
                citation_count=0,
                max_author_citations=0,
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
                arxiv_url=f"https://arxiv.org/abs/{arxiv_id}",
                fields_of_study=["Computer Science"] + extracted_tags,
                source="arXiv.org",
            ))
            
        # If we broke the threshold, stop fetching entirely
        if not oldest_date_in_batch or oldest_date_in_batch < cutoff_date:
            break
            
        start += max_results
        
        # Hard fail-safe: ArXiv API crashes (500 Server Error) when start + max_results > 10000 
        # unless you use OAI-PMH. Since this is just a short-term Scout, gracefully break.
        if start >= 10000:
            logger.info("Reached ArXiv's safe limit (10k preprints). Concluding Stage 1 Scout.")
            break
            
        time.sleep(3) # Comply with arXiv 3-second delay policy
        
    logger.info("Stage 1 ArXiv Scout finished: %d recent papers fetched.", len(results))
    return results

def run_ingestion(
    db_path: str = "data_pipeline/corpus.db",
    max_papers: int = MAX_PAPERS_TO_FETCH,
    incremental: bool = True,
    arxiv_only: bool = False,
    s2_only: bool = False,
    days: int = 15,
) -> int:
    """
    End-to-end ingestion: fetch papers from arXiv (Scout) and/or S2 (Analyzer), upsert into SQLite.
    Returns total papers processed.
    """
    conn = create_db(db_path)
    total = 0

    # Determine what to run
    run_all = not (arxiv_only or s2_only)

    # ----------------------------------------------------
    # Stage 1: The 'Scout' (Instant arXiv fetching)
    # ----------------------------------------------------
    if arxiv_only or run_all:
        try:
            # If not incremental, fetch a large window (e.g. 1 year)
            scout_days = days if incremental else 365
            arxiv_papers = fetch_fresh_arxiv_papers(days=scout_days)
            for i, paper in enumerate(arxiv_papers):
                upsert_paper(conn, paper)
                if i % 1_000 == 0:
                    conn.commit()
            conn.commit()
            logger.info("Stage 1 Complete: %d recent ArXiv papers upserted.", len(arxiv_papers))
            total += len(arxiv_papers)
        except Exception as e:
            logger.error("Stage 1 ArXiv fetching failed: %s", e)

    # ----------------------------------------------------
    # Stage 2: The 'Analyzer' (S2 semantic enrichment)
    # ----------------------------------------------------
    if s2_only or run_all:
        try:
            # For incremental S2, we might restrict the volume or rely on the 90-day filter in fetch_papers_bulk
            papers = fetch_papers_bulk(max_papers=max_papers)
            for i, paper in enumerate(papers):
                upsert_paper(conn, paper)
                if i % 1_000 == 0:
                    conn.commit()
            conn.commit()
            logger.info("Stage 2 Complete: %d enriched papers upserted.", len(papers))
            total += len(papers)
        except Exception as e:
            logger.error("Stage 2 S2 fetching failed: %s", e)
    
    logger.info("Ingestion session complete. SQLite DB updated at %s", db_path)
    conn.close()
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResearchAgent Data Ingestion")
    parser.add_argument("--full", action="store_true", help="Full refresh (longer window)")
    parser.add_argument("--arxiv", action="store_true", help="Only fetch from arXiv API (Scout)")
    parser.add_argument("--s2", action="store_true", help="Only fetch from Semantic Scholar (Analyzer)")
    parser.add_argument("--days", type=int, default=15, help="Days to look back for arXiv Scout (default: 15)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    
    run_ingestion(
        incremental=not args.full,
        arxiv_only=args.arxiv,
        s2_only=args.s2,
        days=args.days
    )
