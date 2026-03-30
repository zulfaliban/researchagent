"""
data_pipeline/build_index.py
Build FAISS (inner-product / cosine) and optional BM25 indexes from corpus.db.

Usage:
    python data_pipeline/build_index.py
"""
import json
import logging
import os
import sqlite3
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import faiss
import numpy as np

logger = logging.getLogger(__name__)

try:
    import bm25s

    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


def load_papers_from_db(db_path: str) -> list:
    """Return all papers from corpus.db as plain dicts (authors decoded from JSON)."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM papers ORDER BY submitted_date DESC"
    ).fetchall()
    conn.close()

    papers = []
    for row in rows:
        d = dict(row)
        d["authors"] = json.loads(d.get("authors") or "[]")
        papers.append(d)
    return papers


def embed_papers(
    papers: list,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> np.ndarray:
    """
    Encode title + abstract with a SentenceTransformer model.
    Returns float32 ndarray of shape (N, dim), L2-normalised.
    """
    from sentence_transformers import SentenceTransformer  # lazy import
    import gc

    model = SentenceTransformer(model_name)
    texts = [
        (p.get("title") or "") + "\n\n" + (p.get("abstract") or "")
        for p in papers
    ]
    vecs = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    
    # Explicitly unload model to free memory for merging
    del model
    gc.collect()
    
    return vecs.astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build an exact inner-product (cosine equivalent for normalised vecs) index.
    """
    idx = faiss.IndexFlatIP(embeddings.shape[1])
    idx.add(embeddings)
    return idx


def save_index(index: faiss.Index, path: str) -> None:
    """Write FAISS index to disk."""
    faiss.write_index(index, path)
    logger.info("FAISS saved: %s (%d vectors)", path, index.ntotal)


def build_bm25_index(papers: list, save_dir: str) -> None:
    """Build and save a BM25 index over title+abstract text."""
    if not HAS_BM25:
        logger.warning("bm25s not installed — skipping BM25 index")
        return

    corpus = [
        (p.get("title") or "") + " " + (p.get("abstract") or "")
        for p in papers
    ]
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(corpus))
    retriever.save(save_dir)
    logger.info("BM25 saved: %s/", save_dir)


def run_index_build(
    db_path: str = "data_pipeline/corpus.db",
    output_dir: str = "data_pipeline",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    force_full: bool = False,
    update_arxiv_ts: bool = False,
    update_s2_ts: bool = False,
) -> None:
    """
    Incremental index build pipeline:
    1. Filter for papers where is_indexed = 0
    2. Load existing FAISS/NPY/JSON artifacts
    3. Embed only new papers
    4. Append to artifacts and save
    5. Update SQLite is_indexed = 1
    6. Rebuild BM25 for all indexed papers
    """
    from data_pipeline.schema import create_db

    os.makedirs(output_dir, exist_ok=True)
    emb_path = os.path.join(output_dir, "embeddings_minilm.npy")
    faiss_path = os.path.join(output_dir, "index_minilm.faiss")
    id_map_path = os.path.join(output_dir, "id_map.json")

    # Use create_db instead of direct connect to ensure columns (is_indexed) are migrated/created
    conn = create_db(db_path)
    conn.row_factory = sqlite3.Row

    if force_full:
        logger.info("Force-full rebuild requested. Resetting is_indexed=0 for all papers.")
        conn.execute("UPDATE papers SET is_indexed = 0")
        conn.commit()

    # 1. Load papers that need indexing
    new_papers_rows = conn.execute("SELECT * FROM papers WHERE is_indexed = 0").fetchall()
    
    if not new_papers_rows:
        logger.info("No new papers to index.")
        conn.close()
        return

    new_papers = []
    for row in new_papers_rows:
        d = dict(row)
        d["authors"] = json.loads(d.get("authors") or "[]")
        new_papers.append(d)

    # 2. Embed new papers FIRST (heavy model in memory)
    logger.info("Embedding %d new papers…", len(new_papers))
    new_embeddings = embed_papers(new_papers, model_name)

    # 3. Reload existing artifacts SECOND (heavy matrix in memory)
    has_artifacts = all(os.path.exists(p) for p in [emb_path, faiss_path, id_map_path])
    
    if has_artifacts and not force_full:
        try:
            # Use mmap_mode='r' to save physical memory on Windows
            existing_embeddings = np.load(emb_path, mmap_mode='r')
            existing_index = faiss.read_index(faiss_path)
            with open(id_map_path, "r", encoding="utf-8") as fh:
                id_map = json.load(fh)
            logger.info("Loaded existing index with %d papers.", existing_index.ntotal)
        except Exception as e:
            logger.warning("Failed to load artifacts: %s. Starting fresh.", e)
            existing_embeddings = None
            existing_index = None
            id_map = {}
    else:
        existing_embeddings = None
        existing_index = None
        id_map = {}
        logger.info("Starting fresh index build.")

    # 4. Merge
    if existing_embeddings is not None:
        # Note: np.vstack will load mmap'd segments temporarily
        all_embeddings = np.vstack([existing_embeddings, new_embeddings])
        existing_index.add(new_embeddings)
        all_index = existing_index
    else:
        all_embeddings = new_embeddings
        all_index = build_faiss_index(new_embeddings)
    
    # Update ID map (append new indices)
    start_idx = len(id_map)
    for i, p in enumerate(new_papers):
        id_map[str(start_idx + i)] = p["arxiv_id"]

    # 5. Save all updated artifacts
    # On Windows, we must delete the mmap object before overwriting the file
    if existing_embeddings is not None:
        del existing_embeddings
        import gc
        gc.collect()

    np.save(emb_path, all_embeddings.astype("float32"))
    faiss.write_index(all_index, faiss_path)
    with open(id_map_path, "w", encoding="utf-8") as fh:
        json.dump(id_map, fh)
    
    # Capture metadata before freeing memory
    faiss_ntotal = all_index.ntotal

    # Free up memory before rebuilding BM25
    del all_embeddings
    del all_index
    gc.collect()
    
    # 6. Update Database
    logger.info("Updating database flags...")
    for p in new_papers:
        conn.execute("UPDATE papers SET is_indexed = 1 WHERE arxiv_id = ?", (p["arxiv_id"],))
    conn.commit()

    # 7. Rebuild BM25
    # Always rebuild from all indexed papers to maintain global IDF statistics
    logger.info("Rebuilding BM25 index for all papers...")
    all_papers_rows = conn.execute("SELECT * FROM papers WHERE is_indexed = 1").fetchall()
    all_papers = [dict(r) for r in all_papers_rows]

    bm25_dir = os.path.join(output_dir, "bm25_index")
    build_bm25_index(all_papers, bm25_dir)

    # 8. Write Step Metadata
    from datetime import datetime
    meta_path = os.path.join(output_dir, "build_meta.json")
    
    existing_meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                existing_meta = json.load(fh)
        except: pass

    last_arxiv = existing_meta.get("last_arxiv_at")
    last_s2 = existing_meta.get("last_s2_at")
    from datetime import timezone
    now_iso = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0).isoformat() + "Z"
    
    meta = {
        "built_at": now_iso,
        "last_arxiv_at": last_arxiv or now_iso,
        "last_s2_at": last_s2 or now_iso,
        "corpus_size": len(all_papers),
        "faiss_ntotal": faiss_ntotal,
        "db_size_bytes": os.path.getsize(db_path),
        "schema_version": 1
    }
    if update_arxiv_ts: meta["last_arxiv_at"] = now_iso
    if update_s2_ts: meta["last_s2_at"] = now_iso

    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("Metadata saved: %s", meta_path)

    conn.close()
    logger.info("Index update complete. Total indexed: %d", len(all_papers))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Force full rebuild")
    parser.add_argument("--update-arxiv", action="store_true", help="Update last_arxiv_at timestamp")
    parser.add_argument("--update-s2", action="store_true", help="Update last_s2_at timestamp")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run_index_build(
        force_full=args.full,
        update_arxiv_ts=args.update_arxiv,
        update_s2_ts=args.update_s2
    )
