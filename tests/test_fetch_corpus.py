from data_pipeline.fetch_corpus import parse_s2_paper


def test_parse_extracts_arxiv_id():
    raw = {
        "paperId": "s2abc",
        "title": "Test",
        "abstract": "Abstract",
        # authors returns only authorId + name from bulk endpoint (no citationCount)
        "authors": [{"authorId": "1", "name": "Alice"}],
        "year": 2024,
        "publicationDate": "2024-01-15",
        "venue": "NeurIPS",
        "citationCount": 10,
        "externalIds": {"ArXiv": "2401.00001"},
        "openAccessPdf": {"url": "https://arxiv.org/pdf/2401.00001"},
        "s2FieldsOfStudy": [{"category": "Computer Science"}],
    }
    paper = parse_s2_paper(raw)
    assert paper.arxiv_id == "2401.00001"
    # max_author_citations is always 0 from bulk endpoint (no authors.citationCount)
    assert paper.max_author_citations == 0


def test_parse_skips_non_arxiv():
    raw = {
        "paperId": "x",
        "title": "T",
        "abstract": "A",
        "authors": [],
        "externalIds": {},
        "citationCount": 0,
    }
    assert parse_s2_paper(raw) is None
