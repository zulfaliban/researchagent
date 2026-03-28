"""
Integration test and latency benchmark for the 3-stage hybrid retrieval pipeline.
"""
import time
import os
import pytest
from app import (
    Paper,
    select_embedding_candidates,
    load_bm25_index,
    load_precomputed_embeddings
)

@pytest.fixture
def dummy_papers():
    # Return a basic list of mock papers matching indices or random
    # We will just load them from the DB for actual testing since we have it indexed.
    import sqlite3
    import json
    if not os.path.exists("data_pipeline/corpus.db"):
        return []
    conn = sqlite3.connect("data_pipeline/corpus.db")
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM papers LIMIT 5000").fetchall()
    papers = []
    for r in rows:
        d = dict(r)
        authors = json.loads(d.get("authors") or "[]")
        p = Paper(
            arxiv_id=d["arxiv_id"],
            title=d["title"],
            authors=authors,
            email_domains=[],
            abstract=d["abstract"] or "",
            submitted_date=d["submitted_date"],
            pdf_url=d["pdf_url"] or "",
            arxiv_url=d["arxiv_url"] or f"https://arxiv.org/abs/{d['arxiv_id']}",
            venue=d["venue"],
            source=d.get("source", "Semantic Scholar")
        )
        papers.append(p)
    return papers

def test_hybrid_retrieval(dummy_papers):
    if not dummy_papers:
        pytest.skip("No corpus DB found.")

    # Benchmark indexing load
    t0 = time.time()
    bm25_retriever, arxiv_to_pos = load_bm25_index()
    embeddings = load_precomputed_embeddings()
    t_load = time.time() - t0
    
    print(f"Artifact Load Time: {t_load:.2f}s")
    
    if bm25_retriever is None or embeddings is None:
        pytest.skip("Index artifacts missing. Run build_index.py first.")
    
    query = "Recommendation systems and matrix factorization"
    
    t0 = time.time()
    candidates = select_embedding_candidates(
        dummy_papers, 
        query, 
        llm_config=None, 
        embedding_model="", 
        provider="", 
        max_candidates=150
    )
    t_retrieval = time.time() - t0
    
    print(f"Retrieval Time (3-stage): {t_retrieval:.2f}s")
    
    assert len(candidates) <= 150
    
    print("Top 5 candidates:")
    for i, c in enumerate(candidates[:5]):
        score = c.semantic_relevance if c.semantic_relevance is not None else 0.0
        print(f" {i+1}. {c.title} ({score:.4f})")
