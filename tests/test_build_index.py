import numpy as np
import faiss
from data_pipeline.build_index import build_faiss_index, save_index


def test_faiss_index_shape():
    vecs = np.random.randn(20, 384).astype("float32")
    faiss.normalize_L2(vecs)
    idx = build_faiss_index(vecs)
    assert idx.ntotal == 20
    D, I = idx.search(vecs[:1], k=5)
    assert I.shape == (1, 5)


def test_save_reload(tmp_path):
    vecs = np.random.randn(10, 384).astype("float32")
    faiss.normalize_L2(vecs)
    idx = build_faiss_index(vecs)
    p = str(tmp_path / "test.faiss")
    save_index(idx, p)
    reloaded = faiss.read_index(p)
    assert reloaded.ntotal == 10
