import sqlite3
import sys

sys.path.insert(0, ".")
from data_pipeline.schema import create_db, PaperRecord


def test_db_creates_table(tmp_path):
    conn = create_db(str(tmp_path / "test.db"))
    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    ]
    assert "papers" in tables


def test_paper_record_fields():
    p = PaperRecord(
        arxiv_id="2301.00001",
        s2_id="abc",
        title="T",
        abstract="A",
        authors=["Alice"],
        submitted_date="2024-01-01",
        venue="NeurIPS",
        citation_count=42,
        max_author_citations=1200,
        pdf_url="https://arxiv.org/pdf/2301.00001",
        arxiv_url="https://arxiv.org/abs/2301.00001",
        fields_of_study=["Computer Science"],
    )
    assert p.citation_count == 42
