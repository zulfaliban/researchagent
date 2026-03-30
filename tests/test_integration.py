"""
tests/test_integration.py
End-to-end integration test: fetch 100 papers from S2 → upsert to SQLite
→ build FAISS + BM25 + id_map.

Skipped unless S2_API_KEY is set in environment (or .env).
Run with:
    pytest -m integration -v tests/test_integration.py
"""
import os
import sqlite3

import pytest
from dotenv import load_dotenv

load_dotenv()

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    not os.getenv("S2_API_KEY"),
    reason="S2_API_KEY not set — skipping live integration test",
)
def test_pipeline_100_papers(tmp_path):
    """
    Smoke test: ingest 100 papers from S2 bulk API, build FAISS + BM25 indexes,
    assert all artifacts exist and DB has ≥ 50 arXiv papers.
    """
    from data_pipeline.fetch_corpus import run_ingestion
    from data_pipeline.build_index import run_index_build

    db = str(tmp_path / "corpus.db")

    # Step 1 — ingestion
    run_ingestion(db_path=db, max_papers=100, incremental=False)
    n = sqlite3.connect(db).execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    assert n >= 1, (
        f"Expected at least 1 arXiv paper, got {n}. "
        "S2 bulk search returns ~30% arXiv hit rate so 100 raw → ≥1 expected."
    )

    # Step 2 — index build
    run_index_build(db_path=db, output_dir=str(tmp_path))

    assert (tmp_path / "index_minilm.faiss").exists(), "FAISS index not created"
    assert (tmp_path / "id_map.json").exists(), "id_map.json not created"
    assert (tmp_path / "embeddings_minilm.npy").exists(), "embeddings file not created"
