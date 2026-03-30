"""
tests/benchmark.py
Performance benchmark for the pre-built corpus index.

Run with:
    python tests/benchmark.py

Pass criteria (from PLAN.md):
    DB row count         < 50 ms
    FAISS load           < 100 ms
    FAISS query k=150    < 5 ms
"""
import sqlite3
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import faiss
import numpy as np

BASE = "data_pipeline"

PASS_CRITERIA = {
    "DB row count": 50.0,
    "FAISS load": 100.0,
    "FAISS query k=150": 5.0,
}


def bench(label: str, fn) -> float:
    t = time.perf_counter()
    result = fn()
    ms = (time.perf_counter() - t) * 1000
    status = "✅" if ms < PASS_CRITERIA.get(label, float("inf")) else "❌"
    print(f"{status}  {label:<30} {ms:>8.1f} ms   result={result}")
    return ms


def main() -> None:
    db_path = Path(BASE) / "corpus.db"
    faiss_path = Path(BASE) / "index_minilm.faiss"

    if not db_path.exists():
        print(f"ERROR: {db_path} not found — run data_pipeline/fetch_corpus.py first.")
        sys.exit(1)
    if not faiss_path.exists():
        print(f"ERROR: {faiss_path} not found — run data_pipeline/build_index.py first.")
        sys.exit(1)

    print(f"\nBenchmark — {BASE}/\n{'=' * 55}")

    results: dict[str, float] = {}

    results["DB row count"] = bench(
        "DB row count",
        lambda: sqlite3.connect(str(db_path))
                         .execute("SELECT COUNT(*) FROM papers")
                         .fetchone()[0],
    )

    idx = faiss.read_index(str(faiss_path))
    results["FAISS load"] = bench("FAISS load", lambda: f"{idx.ntotal} vectors")

    q = np.random.randn(1, idx.d).astype("float32")
    faiss.normalize_L2(q)
    results["FAISS query k=150"] = bench(
        "FAISS query k=150",
        lambda: idx.search(q, 150)[1].shape,
    )

    print(f"\n{'=' * 55}")
    all_pass = all(
        results[k] < PASS_CRITERIA[k]
        for k in PASS_CRITERIA
    )
    if all_pass:
        print("All benchmarks PASSED ✅")
    else:
        failed = [k for k in PASS_CRITERIA if results[k] >= PASS_CRITERIA[k]]
        print(f"FAILED benchmarks: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
