import os
import json
import time
import textwrap
import io
import zipfile
import math
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, date
from typing import List, Optional, Dict, Any
import numpy as np
import pathlib

try:
    import streamlit as st
    import requests
    import feedparser
    import pandas as pd
    from openai import OpenAI, NotFoundError, BadRequestError
    from groq import Groq
except ImportError as e:
    missing = str(e).split("'")[1]
    print(f"Missing package: {missing}")
    print("Please run: pip install streamlit requests feedparser openai pandas groq")
    raise

# Optional local embedding model for free mode
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:
    SentenceTransformer = None  # type: ignore

# Optional Google Gemini client
try:
    from google import genai  # type: ignore
    from google.genai import types # type: ignore
except ImportError:
    genai = None  # type: ignore

try:
    import bm25s
except ImportError:
    bm25s = None

# =========================
# Constants
# =========================

MIN_FOR_PREDICTION = 20
OPENAI_EMBEDDING_MODEL_NAME = "text-embedding-3-large"
GEMINI_EMBEDDING_MODEL_NAME = "gemini-embedding-001"

# MONEYBALL DEFAULTS
DEFAULT_MONEYBALL_WEIGHTS = {
    "weight_fame": 0.84,
    "weight_hype": 0.0,
    "weight_sniper": 0.0,
    "weight_utility": 0.16
}



# =========================
# Data structures
# =========================

@dataclass
class LLMConfig:
    api_key: str
    model: str
    api_base: Optional[str]
    provider: str = "openai"  # "openai", "gemini", "groq", or "free_local"


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: List[str]
    email_domains: List[str]
    abstract: str
    submitted_date: datetime
    pdf_url: str
    arxiv_url: str
    predicted_citations: Optional[float] = None
    prediction_explanations: Optional[List[str]] = None
    semantic_relevance: Optional[float] = None
    semantic_reason: Optional[str] = None
    focus_label: Optional[str] = None
    llm_relevance_score: Optional[float] = None
    venue: Optional[str] = None
    source: Optional[str] = None


# =========================
# Utility functions
# =========================

def get_date_range(option: str) -> (date, date):
    today = date.today()
    if option == "Last 3 Days":
        return today - timedelta(days=3), today
    elif option == "Last Week":
        return today - timedelta(days=7), today
    elif option == "Last Month":
        return today - timedelta(days=30), today
    else:
        raise ValueError(f"Unknown date range option: {option}")


def ensure_folder(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


def save_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def get_corpus_dir() -> pathlib.Path:
    """Returns the writable directory for data pipeline artifacts."""
    local_dir = pathlib.Path("data_pipeline")
    # In Streamlit Cloud and most containers, /tmp is always writable.
    # Check if we are running in a deployed environment or if local data_pipeline is missing/not writable.
    is_streamlit_cloud = os.getenv("STREAMLIT_SHARING_MODE") is not None
    if is_streamlit_cloud or not local_dir.exists():
        return pathlib.Path("/tmp/data_pipeline")
    return local_dir


def _check_corpus_freshness():
    """Lightweight per-search freshness check that reloads data in-place without restart."""
    import time
    import boto3
    from botocore.exceptions import ClientError

    # 1. Throttle: check R2 at most every 30 minutes per session to save egress/API calls
    now = time.time()
    last_check = st.session_state.get("_freshness_checked_at", 0)
    if now - last_check < 1800: # 1800s = 30 min
        return

    # 2. Get credentials
    access_key = st.secrets.get("R2_ACCESS_KEY_ID") or os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = st.secrets.get("R2_SECRET_ACCESS_KEY") or os.environ.get("R2_SECRET_ACCESS_KEY")
    endpoint = st.secrets.get("R2_ENDPOINT") or os.environ.get("R2_ENDPOINT")
    bucket_name = st.secrets.get("R2_BUCKET") or os.environ.get("R2_BUCKET")

    if not all([access_key, secret_key, endpoint, bucket_name]):
        return

    try:
        s3 = boto3.client(
            's3',
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        
        # 3. HEAD request to check ETag of the metadata file
        response = s3.head_object(Bucket=bucket_name, Key="corpus/build_meta.json")
        new_etag = response.get('ETag', '').strip('"')
        old_etag = st.session_state.get("_corpus_etag")

        if old_etag and new_etag == old_etag:
            st.session_state["_freshness_checked_at"] = now
            return

        # 4. If ETag differs, refresh cached resources
        if old_etag:
            # We only show this if it's an actual update detected since session start
            msg = st.info("New corpus data available — refreshing index (this takes ~20 seconds)...")
            st.cache_resource.clear()
            download_corpus_artifacts()
            msg.empty()

        st.session_state["_corpus_etag"] = new_etag
        st.session_state["_freshness_checked_at"] = now

    except Exception:
        # Fail silently: data refresh is non-critical, we don't want to block the search
        pass


@st.cache_resource(show_spinner=False)
def download_corpus_artifacts():
    """
    Startup sync from Cloudflare R2 with freshness check.
    Syncs artifacts down to the local get_corpus_dir() if remote is newer.
    """
    import boto3
    from concurrent.futures import ThreadPoolExecutor

    # 1. Resolve credentials
    # Use st.secrets if in Streamlit Cloud, otherwise environment variables
    key_id = st.secrets.get("R2_ACCESS_KEY_ID") or os.getenv("R2_ACCESS_KEY_ID")
    access_key = st.secrets.get("R2_SECRET_ACCESS_KEY") or os.getenv("R2_SECRET_ACCESS_KEY")
    endpoint = st.secrets.get("R2_ENDPOINT") or os.getenv("R2_ENDPOINT")
    bucket = st.secrets.get("R2_BUCKET") or os.getenv("R2_BUCKET")

    if not all([key_id, access_key, endpoint, bucket]):
        st.info("💡 R2 credentials not fully set; skipping remote corpus sync.")
        return

    corpus_dir = get_corpus_dir()
    corpus_dir.mkdir(parents=True, exist_ok=True)
    meta_path = corpus_dir / "build_meta.json"
    status_placeholder = st.empty()
    try:
        s2_client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=key_id,
            aws_secret_access_key=access_key,
        )

        # 2. Freshness Check
        status_placeholder.info("Checking for fresh corpus updates on R2...")
        local_meta = {}
        if meta_path.exists():
            try:
                local_meta = json.load(open(meta_path, "r", encoding="utf-8"))
            except:
                pass

        # Download remote build_meta.json to a temp buffer
        remote_meta_obj = s2_client.get_object(Bucket=bucket, Key="corpus/build_meta.json")
        remote_meta = json.loads(remote_meta_obj["Body"].read().decode("utf-8"))

        remote_ts = remote_meta.get("built_at", "")
        local_ts = local_meta.get("built_at", "")
        remote_ver = remote_meta.get("schema_version", 1)
        local_ver = local_meta.get("schema_version", 0)

        if remote_ts == local_ts and remote_ver == local_ver:
            # No update needed
            return

        # 3. Full Download
        status_placeholder.info("🔄 Downloading fresh corpus artifacts from R2...")
        
        # Files to download (prefix 'corpus/' removed from bucket key during sync to app local)
        # Note: scheduler.py pushes with prefix 'corpus/'. 
        files = [
            "corpus.db",
            "index_minilm.faiss",
            "embeddings_minilm.npy",
            "id_map.json",
            "build_meta.json"
        ]
        
        def _download(filename):
            dest = corpus_dir / filename
            s2_client.download_file(bucket, f"corpus/{filename}", str(dest))
            
        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(_download, files)

        # 4. BM25 special case (recursive directory)
        # We'll just download the index files manually for now
        # bm25s keeps files in bm25_index/ folder
        # For simplicity, if we need full directory sync, we'd list the bucket
        # but here we'll just download the known ones
        bm25_dir = corpus_dir / "bm25_index"
        bm25_dir.mkdir(exist_ok=True)
        
        # We need to list objects in corpus/bm25_index/
        response = s2_client.list_objects_v2(Bucket=bucket, Prefix="corpus/bm25_index/")
        for obj in response.get("Contents", []):
            key = obj["Key"]
            filename = key.replace("corpus/bm25_index/", "")
            if filename:
                s2_client.download_file(bucket, key, str(bm25_dir / filename))

        status_placeholder.success("✅ Corpus artifacts updated successfully!")
        time.sleep(1)
        status_placeholder.empty()

    except Exception as e:
        status_placeholder.warning(f"⚠️ Remote corpus sync failed: {e}. Falling back to local data.")


def build_query_brief(research_brief: str, not_looking_for: str) -> str:
    research_brief = research_brief.strip()
    not_looking_for = not_looking_for.strip()
    parts = []
    if research_brief:
        parts.append("RESEARCH BRIEF:\n" + research_brief)
    if not_looking_for:
        parts.append("WHAT I AM NOT LOOKING FOR:\n" + not_looking_for)
    if not parts:
        return "The user did not provide any research brief."
    return "\n\n".join(parts)


def parse_not_terms(not_text: str) -> List[str]:
    not_text = not_text.strip()
    if not not_text:
        return []
    parts = re.split(r"[,\n;]+", not_text)
    terms = [p.strip().lower() for p in parts if p.strip()]
    return terms


def filter_papers_by_not_terms(papers: List[Paper], not_text: str) -> (List[Paper], int):
    terms = parse_not_terms(not_text)
    if not terms or not papers:
        return papers, 0

    filtered: List[Paper] = []
    removed = 0
    for p in papers:
        text = f"{p.title} {p.abstract}".lower()
        if any(term in text for term in terms):
            removed += 1
        else:
            filtered.append(p)

    return filtered, removed


def filter_papers_by_venue(
    papers: List[Paper],
    venue_filter_type: str,
    selected_category: Optional[str],
    selected_venues: List[str]
):
    if venue_filter_type == "None":
        return papers

    filtered = []
    for p in papers:
        venue = (p.venue or "").lower()

        if venue_filter_type == "All Conferences":
            if any(conf.lower() in venue for conf in CONFERENCE_KEYWORDS):
                filtered.append(p)

        elif venue_filter_type == "All Journals":
            if any(j.lower() in venue for j in JOURNAL_KEYWORDS):
                filtered.append(p)

        elif venue_filter_type == "Specific Venue":
            if selected_category == "Conference":
                if selected_venues and any(sel.lower() in venue for sel in selected_venues):
                    filtered.append(p)
            elif selected_category == "Journal":
                if selected_venues and any(sel.lower() in venue for sel in selected_venues):
                    filtered.append(p)

    return filtered


# =========================
# Venue extraction helpers
# =========================

CONFERENCE_KEYWORDS = [
    "EMNLP", "ACL", "NAACL", "EACL",
    "NeurIPS", "ICLR", "ICML",
    "CVPR", "ECCV",
    "ICASSP", "AAAI", "AISTATS",
]

JOURNAL_KEYWORDS = [
    "Nature", "Science",
    "JMLR", "Journal of Machine Learning Research",
    "TPAMI", "IEEE Transactions on Pattern Analysis",
    "Artificial Intelligence Journal",
    "IJCV", "International Journal of Computer Vision",
    "Nature Machine Intelligence", "Nature Communications",
]

NEGATIVE_VENUE_SIGNALS = ["submitted to", "under review", "preprint"]

ARXIV_CATEGORIES: Dict[str, List[str]] = {
    # Keep it practical; you can extend anytime
    "Computer Science": [
        "cs.AI", "cs.LG", "cs.HC", "cs.CL", "cs.CV", "cs.RO", "cs.IR", "cs.NE", "cs.SE",
        "cs.CR", "cs.DS", "cs.DB", "cs.SI", "cs.MM", "cs.IT", "cs.PF", "cs.MA",
    ],
    # "Statistics": ["stat.ML", "stat.AP", "stat.CO", "stat.TH"],
    # "Mathematics": ["math.OC", "math.ST", "math.IT", "math.PR", "math.NA"],
    # "Physics": ["physics.comp-ph", "physics.data-an", "physics.soc-ph", "physics.optics"],
    # "Quantitative Biology": ["q-bio.QM", "q-bio.NC", "q-bio.BM"],
    # "Quantitative Finance": ["q-fin.MF", "q-fin.ST", "q-fin.CP", "q-fin.TR"],
    # "Electrical Engineering and Systems Science": ["eess.IV", "eess.SP", "eess.SY", "eess.AS"],
    # "Economics": ["econ.EM", "econ.TH"],
}

ARXIV_CODE_TO_NAME = {
    "cs.AI": "Artificial Intelligence",
    "cs.LG": "Machine Learning",
    "cs.HC": "Human-Computer Interaction",
    "cs.CL": "Computation and Language",
    "cs.CV": "Computer Vision and Pattern Recognition",
    "cs.RO": "Robotics",
    "cs.IR": "Information Retrieval",
    "cs.NE": "Neural and Evolutionary Computing",
    "cs.SE": "Software Engineering",
    "cs.CR": "Cryptography and Security",
    "cs.DS": "Data Structures and Algorithms",
    "cs.DB": "Databases",
    "cs.SI": "Social and Information Networks",
    "cs.MM": "Multimedia",
    "cs.IT": "Information Theory",
    "cs.PF": "Performance",
    "cs.MA": "Multiagent Systems",
}

def extract_venue(comment: str) -> Optional[str]:
    if not comment:
        return None
    c = comment.lower()
    if any(sig in c for sig in NEGATIVE_VENUE_SIGNALS):
        return None
    all_venues = sorted(CONFERENCE_KEYWORDS + JOURNAL_KEYWORDS)
    for venue in all_venues:
        if venue.lower() in c:
            return venue
    return None

def build_arxiv_category_query(
    main_category: str,
    subcategories: List[str],
) -> str:
    """
    Returns arXiv API query fragment like:
      (cat:cs.AI OR cat:cs.LG OR cat:stat.ML)
    If subcategories empty, falls back to all subcats in main_category.
    If main_category == "All", uses all subcategories across all mains.
    """
    if main_category == "All":
        cats = sorted({c for subs in ARXIV_CATEGORIES.values() for c in subs})
    else:
        cats = subcategories if subcategories else ARXIV_CATEGORIES.get(main_category, [])

    # Safety fallback (so your app never crashes)
    if not cats:
        cats = ["cs.AI", "cs.LG", "cs.HC"]

    return "(" + " OR ".join([f"cat:{c}" for c in cats]) + ")"

def fetch_papers_from_db(
    start_date: date,
    end_date: date,
    category_filter: Optional[str] = None,
    subcats: Optional[List[str]] = None
) -> List[Paper]:
    """
    Fetch papers from the local SQLite 20k corpus.
    Filters by submitted_date range, and optionally checks abstract/title
    for subcategory keyword matches to simulate arXiv category filtering.
    """
    import os
    import json
    import sqlite3 as _sq
    db_path = get_corpus_dir() / "corpus.db"
    if not db_path.exists():
        return []

    conn = _sq.connect(db_path, check_same_thread=False)
    conn.row_factory = _sq.Row
    
    query = "SELECT * FROM papers WHERE date(submitted_date) >= date(?) AND date(submitted_date) <= date(?)"
    params = [start_date.isoformat(), end_date.isoformat()]
    
    if category_filter and category_filter != "All":
        query += " AND fields_of_study LIKE ?"
        params.append(f"%{category_filter}%")

    if subcats and category_filter != "All":
        or_clauses = []
        for cat_code in subcats:
            cat_name = ARXIV_CODE_TO_NAME.get(cat_code, cat_code)
            # Remove minor stop words for better partial match
            words = [w for w in cat_name.split() if w.lower() not in ('and', 'or', 'of')]
            if words:
                keyword = words[0]
                if len(words) > 1:
                    keyword = " ".join(words[:2]) # e.g. "Artificial Intelligence"
                
                # Check for either standard keyword simulation OR absolute 100% precision raw arXiv tag
                or_clauses.append("(title LIKE ? OR abstract LIKE ? OR fields_of_study LIKE ?)")
                params.extend([f"%{keyword}%", f"%{keyword}%", f"%{cat_code}%"])
                
        if or_clauses:
            query += " AND (" + " OR ".join(or_clauses) + ")"
        
    rows = conn.execute(query, params).fetchall()

    papers: List[Paper] = []
    for r in rows:
        d = dict(r)
        date_str = d.get("submitted_date", "2024-01-01")
        if "T" not in date_str:
            date_str = date_str + "T00:00:00"
            
        papers.append(Paper(
            arxiv_id=d["arxiv_id"],
            title=d["title"],
            authors=json.loads(d.get("authors") or "[]"),
            email_domains=[],
            abstract=d.get("abstract") or "",
            submitted_date=datetime.fromisoformat(date_str.replace("Z", "+00:00")),
            pdf_url=d.get("pdf_url") or "",
            arxiv_url=d.get("arxiv_url") or "",
            venue=d.get("venue"),
            source=d.get("source"),
        ))
    return papers


# =========================
# Generic LLM call + JSON helper
# =========================

def call_llm(prompt: str, llm_config: LLMConfig, label: str = "") -> str:
    if "last_prompts" not in st.session_state:
        st.session_state["last_prompts"] = {}
    st.session_state["last_prompts"][label or "default"] = prompt

    max_retries = 3
    for attempt in range(max_retries):
        try:
            if llm_config.provider == "openai":
                # Standard OpenAI init
                client_args = {"api_key": llm_config.api_key}
                if llm_config.api_base and llm_config.api_base.strip():
                    client_args["base_url"] = llm_config.api_base
                
                client = OpenAI(**client_args)
                
                messages = [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt},
                ]
                kwargs: Dict[str, Any] = {"model": llm_config.model, "messages": messages}
                
                if not (llm_config.model.startswith("o1") or llm_config.model.startswith("gpt-5")):
                    kwargs["temperature"] = 0.2
                
                resp = client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content

            elif llm_config.provider == "gemini":
                if genai is None:
                    st.error("Gemini provider selected but google-genai package is not installed.")
                    st.stop()
                client = genai.Client(api_key=llm_config.api_key)
                response = client.models.generate_content(
                    model=llm_config.model,
                    contents=prompt,
                )
                
                # Handle mixed content
                if hasattr(response, 'candidates') and response.candidates:
                    cand = response.candidates[0]
                    if hasattr(cand, 'content') and hasattr(cand.content, 'parts'):
                        texts = []
                        for part in cand.content.parts:
                            if hasattr(part, 'text') and part.text:
                                texts.append(part.text)
                        if texts:
                            return "".join(texts)
                return getattr(response, 'text', "")

            elif llm_config.provider == "groq":
                client = Groq(api_key=llm_config.api_key)
                response = client.chat.completions.create(
                    model=llm_config.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )
                return response.choices[0].message.content

            else:
                raise ValueError(f"Unknown provider: {llm_config.provider}")

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"LLM call failed ({label}): {e}")
                raise e
            time.sleep(2 * (attempt + 1))


def safe_parse_json_array(raw: str) -> Optional[List[Dict[str, Any]]]:
    if not raw or not raw.strip(): return None
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Try finding brackets
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list): return parsed
        except: pass

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list): return parsed
    except: return None
    return None


# =========================
# Embeddings
# =========================

def embed_texts_openai(texts: List[str], llm_config: LLMConfig, embedding_model: str) -> List[List[float]]:
    if not texts: return []
    
    # Robust Init with Retries
    client_args = {"api_key": llm_config.api_key}
    if llm_config.api_base and llm_config.api_base.strip():
        client_args["base_url"] = llm_config.api_base
    client = OpenAI(**client_args)
    
    all_embeddings: List[List[float]] = []
    batch_size = 100
    
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = client.embeddings.create(model=embedding_model, input=batch)
                for d in resp.data:
                    all_embeddings.append(d.embedding)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    st.error(f"Embedding API call failed: {e}")
                    raise
                time.sleep(2 * (attempt + 1))
                
    return all_embeddings


def embed_texts_gemini(texts: List[str], llm_config: LLMConfig, embedding_model: str) -> List[List[float]]:
    if not texts: return []
    if genai is None:
        st.error("google-genai package missing.")
        st.stop()
    client = genai.Client(api_key=llm_config.api_key)
    all_embeddings: List[List[float]] = []
    batch_size = 100
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        try:
            response = client.models.embed_content(model=embedding_model, contents=batch)
        except Exception as e:
            st.error(f"Gemini embedding API call failed: {e}")
            raise
        for emb in getattr(response, "embeddings", []):
            all_embeddings.append(list(emb.values))
    return all_embeddings


@st.cache_resource(show_spinner=False)
def get_local_embed_model() -> SentenceTransformer:
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed.")
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")



def embed_texts_local(texts: List[str]) -> List[List[float]]:
    if not texts: return []
    try:
        model = get_local_embed_model()
    except Exception as e:
        st.error(f"Local embedding model error: {e}")
        st.stop()
    vectors = model.encode(texts, convert_to_numpy=True)
    return vectors.tolist()


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    if not vec1 or not vec2 or len(vec1) != len(vec2): return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1)
    norm2 = sum(b * b for b in vec2)
    if norm1 == 0.0 or norm2 == 0.0: return 0.0
    return dot / (math.sqrt(norm1) * math.sqrt(norm2))


@st.cache_resource(show_spinner=False)
def load_bm25_index():
    if not bm25s:
        return None, None
    base = get_corpus_dir()
    bm25_path = base / "bm25_index"
    id_map_path = base / "id_map.json"
    if not bm25_path.exists() or not id_map_path.exists():
        return None, None
    try:
        retriever = bm25s.BM25.load(str(bm25_path))
        with open(id_map_path, "r", encoding="utf-8") as f:
            id_map = json.load(f)
        arxiv_to_pos = {v: int(k) for k, v in id_map.items()}
        return retriever, arxiv_to_pos
    except Exception as e:
        print(f"Failed to load BM25 index: {e}")
        return None, None

def bm25_recall(papers: List[Paper], query_brief: str, retriever, arxiv_to_pos: dict, n1: int = 600) -> List[Paper]:
    if not retriever or not papers:
        return papers
    # Build dictionary for O(1) fetch
    paper_dict = {p.arxiv_id: p for p in papers}
    
    tokens = bm25s.tokenize([query_brief])
    try:
        res, scores = retriever.retrieve(tokens, k=n1)
    except Exception as e:
        print(f"BM25 retrieve error: {e}")
        return papers
    
    positions = res[0]
    
    # Needs id_map inverted to map position back to arxiv_id. 
    # But wait, we inverted the id_map to arxiv_to_pos. So we need pos_to_arxiv.
    # Let's recreate pos_to_arxiv locally
    pos_to_arxiv = {v: k for k, v in arxiv_to_pos.items()}
    
    recalled_papers = []
    seen = set()
    for pos in positions:
        pos_int = int(pos)
        arxiv_id = pos_to_arxiv.get(pos_int)
        if arxiv_id and arxiv_id in paper_dict and arxiv_id not in seen:
            recalled_papers.append(paper_dict[arxiv_id])
            seen.add(arxiv_id)
            
    if len(recalled_papers) < 50:
        st.info("BM25 recall found fewer than 50 intersecting papers. Skiping strict BM25 pruning.")
        return papers
    
    return recalled_papers

@st.cache_resource(show_spinner=False)
def load_precomputed_embeddings():
    base = get_corpus_dir()
    emb_path = base / "embeddings_minilm.npy"
    if not emb_path.exists():
        return None
    try:
        return np.load(str(emb_path), mmap_mode='r')
    except Exception as e:
        print(f"Failed to load precomputed embeddings: {e}")
        return None

def minilm_vector_rerank(papers: List[Paper], query_brief: str, embeddings: Optional[np.ndarray], arxiv_to_pos: dict, n2: int = 300) -> List[Paper]:
    if not papers: return []
    if embeddings is None or not arxiv_to_pos:
        # Fallback to runtime embed
        texts = [p.title + "\n\n" + p.abstract for p in papers]
        paper_vecs = embed_texts_local(texts)
        q_vec = embed_texts_local([query_brief])[0]
        scored = []
        for p, vec in zip(papers, paper_vecs):
            sim = cosine_similarity(q_vec, vec)
            p.semantic_relevance = sim
            scored.append((sim, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        k = min(n2, len(scored))
        return [p for _, p in scored[:k]]
        
    try:
        model = get_local_embed_model()
        q_vec = model.encode([query_brief], normalize_embeddings=True)[0]
    except Exception as e:
        print(f"Local query embed error: {e}")
        return papers

    scored = []
    # Collect indices for batch extraction if possible, or dot product individually
    for p in papers:
        pos = arxiv_to_pos.get(p.arxiv_id)
        if pos is not None and pos < embeddings.shape[0]:
            vec = embeddings[pos]
            sim = float(np.dot(vec, q_vec))
        else:
            sim = 0.0 # Paper not in index
        p.semantic_relevance = sim
        scored.append((sim, p))
        
    scored.sort(key=lambda x: x[0], reverse=True)
    k = min(n2, len(scored))
    return [p for _, p in scored[:k]]

@st.cache_resource(show_spinner=False)
def get_cross_encoder_model():
    try:
        from sentence_transformers import CrossEncoder
        st.info("Loading CrossEncoder model for precision re-ranking (first run only)…")
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception as e:
        print(f"CrossEncoder load error: {e}")
        return None

def cross_encoder_rerank(papers: List[Paper], query_brief: str, n3: int = 150) -> List[Paper]:
    if not papers: return []
    model = get_cross_encoder_model()
    if not model:
        # Fallback, just return top-N of whatever they came in as
        return papers[:n3]
        
    pairs = [[query_brief, p.title + "\n\n" + p.abstract] for p in papers]
    try:
        scores = model.predict(pairs)
        # normalize to 0-1 range roughly using sigmoid if we want it to look like cosine, 
        # or just sort by raw logits for reranking.
        # We will sort by raw logit, but we can set semantic_relevance to an arbitrary 0-1 scale.
        
        scored = []
        for p, score in zip(papers, scores):
            score_float = float(score)
            # Sigmoid normalization for relevance
            p.semantic_relevance = 1 / (1 + math.exp(-score_float)) 
            scored.append((score_float, p))
            
        scored.sort(key=lambda x: x[0], reverse=True)
        k = min(n3, len(scored))
        return [p for _, p in scored[:k]]
    except Exception as e:
        print(f"CrossEncoder predict error: {e}")
        return papers[:n3]


def select_embedding_candidates(
    papers: List[Paper],
    query_brief: str,
    llm_config: Optional[LLMConfig] = None,
    embedding_model: str = "",
    provider: str = "",
    max_candidates: int = 150,
) -> List[Paper]:
    """
    3-Stage Hybrid Search:
    Stage 1: BM25 Lexical Recall (narrow down to Top 600)
    Stage 2: MiniLM Vector Lookup (narrow down to Top 300)
    Stage 3: CrossEncoder Precision Rerank (narrow down to Top 150)
    """
    if not papers: return []
    
    st.write(f"Starting 3-stage hybrid search from {len(papers)} SQLite candidates...")

    # Load artifacts
    bm25_retriever, arxiv_to_pos = load_bm25_index()
    embeddings = load_precomputed_embeddings()

    # Stage 1: BM25 Recall
    if bm25_retriever and arxiv_to_pos:
        st.write("⏱ Stage 1: BM25 lexical recall...")
        stage1_papers = bm25_recall(papers, query_brief, bm25_retriever, arxiv_to_pos, n1=600)
        st.write(f"✅ BM25 selected {len(stage1_papers)} candidates.")
    else:
        st.warning("BM25 index not found. Skipping Stage 1.")
        stage1_papers = papers

    # Stage 2: MiniLM Semantic Filter
    st.write("⏱ Stage 2: MiniLM vector semantic filter...")
    stage2_papers = minilm_vector_rerank(stage1_papers, query_brief, embeddings, arxiv_to_pos if arxiv_to_pos else {}, n2=300)
    st.write(f"✅ Vector reranking selected {len(stage2_papers)} candidates.")
    
    # Stage 3: CrossEncoder Precision Rerank
    st.write("⏱ Stage 3: Cross-Encoder precision reranking...")
    stage3_papers = cross_encoder_rerank(stage2_papers, query_brief, n3=max_candidates)
    st.write(f"✅ Cross-Encoder selected {len(stage3_papers)} final candidates.")

    return stage3_papers


# =========================
# LLM relevance classification
# =========================

def classify_papers_with_llm(
    papers: List[Paper],
    query_brief: str,
    llm_config: LLMConfig,
    batch_size: int = 15,
) -> List[Paper]:
    if not papers: return papers

    for batch_start in range(0, len(papers), batch_size):
        batch = papers[batch_start:batch_start + batch_size]
        paper_blocks = []
        for idx, p in enumerate(batch):
            block = textwrap.dedent(f"""
            Paper {idx}:
            Title: {p.title}
            Abstract: {p.abstract}
            """).strip()
            paper_blocks.append(block)

        instruction = textwrap.dedent(f"""
        You are given a user's research brief and a small set of papers.
        Brief: \"\"\"{query_brief}\"\"\"

        For each paper, decide:
          1. focus_label: "primary", "secondary", or "off-topic".
          2. relevance_score: float 0.0-1.0.
          3. reason: 1-2 sentence explanation.

        Return JSON array:
          [{{ "index": <int>, "focus_label": "...", "relevance_score": <float>, "reason": "..." }}]
        """).strip()

        prompt = "\n\n".join([instruction, "PAPERS:", *paper_blocks])
        
        # Retry logic handled inside call_llm now
        try:
            raw = call_llm(prompt, llm_config, label="classification")
        except:
            raw = ""

        parsed = safe_parse_json_array(raw)

        if parsed is None:
            # st.error("Failed to parse classification JSON.")
            continue

        idx_to_info = {}
        for item in parsed:
            try:
                idx = int(item["index"])
                label = str(item.get("focus_label", "")).strip().lower()
                if label not in ["primary", "secondary", "off-topic"]: label = "off-topic"
                idx_to_info[idx] = {
                    "focus_label": label,
                    "relevance_score": float(item.get("relevance_score", 0.0)),
                    "reason": str(item.get("reason", "")).strip(),
                }
            except: continue

        for idx, p in enumerate(batch):
            info = idx_to_info.get(idx)
            if info:
                p.focus_label = info["focus_label"]
                p.llm_relevance_score = info["relevance_score"]
                p.semantic_reason = info["reason"]
            else:
                p.focus_label = "off-topic"
                p.llm_relevance_score = 0.0

    return papers


def heuristic_classify_papers_free(candidates: List[Paper]) -> List[Paper]:
    if not candidates: return candidates
    ranked = sorted(candidates, key=lambda p: p.semantic_relevance or 0.0, reverse=True)
    n = len(ranked)
    if n == 0: return ranked
    top_k = max(1, min(n, max(10, int(0.3 * n))))
    for idx, p in enumerate(ranked):
        p.llm_relevance_score = p.semantic_relevance or 0.0
        p.focus_label = "primary" if idx < top_k else "secondary"
        if p.semantic_reason is None:
            p.semantic_reason = "Heuristic classification based on embedding similarity."
    return ranked


# =========================
# MONEYBALL Impact Scoring
# =========================

def get_s2_citation_stats(paper: Paper, api_key: Optional[str] = None) -> int:
    """Queries Semantic Scholar to find the max author citations."""
    headers = {"x-api-key": api_key} if api_key else {}
    max_retries = 2 # 3 attempts total (initial + 2 retries)

    def fetch(url, params):
        for attempt in range(max_retries + 1):
            try:
                # INCREASED TIMEOUT TO 10
                r = requests.get(url, headers=headers, params=params, timeout=10)
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 429: # Rate limit, backoff
                     time.sleep(2 * (attempt + 1))
                     continue
            except:
                 if attempt < max_retries:
                     time.sleep(1)
                     continue
        return None
    
    # 1. Try Lookup by ArXiv ID first (More reliable)
    if paper.arxiv_id:
        clean_id = paper.arxiv_id.split('v')[0]
        url = f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{clean_id}"
        params = {"fields": "authors.citationCount"}
        data = fetch(url, params)
        if data:
            auth_cites = [a.get('citationCount', 0) for a in data.get('authors', []) if a.get('citationCount')]
            if auth_cites: return max(auth_cites)

    # 2. Fallback to Title Search
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": paper.title, "limit": 1, "fields": "title,citationCount,authors.citationCount"}
    data = fetch(url, params)
    if data and data.get('data'):
         paper_data = data['data'][0]
         auth_cites = [a.get('citationCount', 0) for a in paper_data.get('authors', []) if a.get('citationCount')]
         return max(auth_cites) if auth_cites else 0

    return 0

def predict_citations_direct(target_papers: List[Paper], llm_config: LLMConfig, batch_size: int = 8) -> List[Paper]:
    """MONEYBALL PREDICTOR: Hybrid Author Data + Custom LLM Narrative."""
    if not target_papers: return target_papers

    weights = DEFAULT_MONEYBALL_WEIGHTS
    if os.path.exists("moneyball_weights.json"):
        try:
            with open("moneyball_weights.json", "r") as f: weights = json.load(f)
        except: pass
    
    s2_key = os.getenv("S2_API_KEY") 
    progress_bar = st.progress(0)
    
    for i, p in enumerate(target_papers):
        # 1. Get Hard Data (Fame Signal)
        max_auth_cites = get_s2_citation_stats(p, s2_key)
        
        # LOGIC: Check if we have data or if the paper is too new
        is_fresh = False
        try:
            # DETERMINISTIC CHECK: Purely based on date, ignoring API flakiness result
            days_old = (datetime.now().date() - p.submitted_date.date()).days
            if days_old <= 5: is_fresh = True
        except: pass

        if max_auth_cites > 0:
            # We have real data -> Standard Model
            h1_fame = min(math.log(max_auth_cites + 1) * 8, 95)
            fame_label = "real"
        elif is_fresh:
            # Too New + No Data -> DO NOT SCORE
            fame_label = "too_new"
            h1_fame = 0.0 # Sentinel
        else:
            # Old + No Data -> Likely obscure
            h1_fame = 0.0
            fame_label = "none"

        if not s2_key: time.sleep(0.3) 
        
        # 2. Calculate Hype/Sniper (Python Heuristics)
        t_lower = p.title.lower()
        h2_hype = 0
        if "benchmark" in t_lower or "dataset" in t_lower: h2_hype += 50
        if "survey" in t_lower: h2_hype += 40
        if "llm" in t_lower: h2_hype += 10
        
        h3_sniper = 0
        if "benchmark" in t_lower: h3_sniper += 50
        niche = ["lidar", "3d", "audio", "wireless", "agriculture", "traffic", "physics"]
        if any(n in t_lower for n in niche): h3_sniper -= 20
        
        # 3. LLM Call: Get Score AND Custom Narrative
        prompt = textwrap.dedent(f"""
            Analyze this abstract.
            1. Rate 'Citation Potential' (0-10) based on market fit (Broad/Hot = High, Niche = Low).
            2. Write 2 short, plain English sentences explaining the score.
               - Sentence 1 (Market Fit): Why is this topic hot or niche? (Do NOT start with "Market Fit:")
               - Sentence 2 (Contribution): What is the specific value? (Do NOT start with "Contribution:")
            
            Title: {p.title}
            Abstract: {p.abstract[:800]}...

            Return JSON: {{ "score": <int>, "bullets": ["string", "string"] }}
        """)
        
        h4_utility = 50.0
        content_bullets = [
            "The topic appears relevant to current research trends.",
            "The paper proposes a specific contribution to the field."
        ]
        
        try:
            raw = call_llm(prompt, llm_config, label="moneyball_narrative")
            if raw:
                if "```" in raw:
                    parts = raw.split("```json")
                    if len(parts) > 1: raw = parts[1].split("```")[0]
                    else: raw = raw.split("```")[1].split("```")[0]
                
                parsed = json.loads(raw.strip())
                h4_utility = float(parsed.get("score", 5) * 10)
                
                if "bullets" in parsed and isinstance(parsed["bullets"], list):
                    raw_list = parsed["bullets"]
                    cleaned_list = []
                    for b in raw_list:
                        b = b.replace("Market Fit:", "").replace("Contribution:", "").strip()
                        cleaned_list.append(b)
                    content_bullets = cleaned_list[:2]
        except: pass

        # 4. Calculate Final Score
        if fame_label == "too_new":
            # SENTINEL VALUE: -1.0 means "Unrated"
            p.predicted_citations = -1.0
        else:
            score = (h1_fame * weights['weight_fame'] + 
                     h2_hype * weights['weight_hype'] + 
                     h3_sniper * weights['weight_sniper'] + 
                     h4_utility * weights['weight_utility'])
            p.predicted_citations = score

        # 5. Construct Final Narrative (3 Bullets)
        final_bullets = []
        
        # Bullet 1: Author Context
        if fame_label == "real":
            if max_auth_cites > 3000:
                final_bullets.append("🚀 **Distribution:** High influence author/lab.")
            elif max_auth_cites > 500:
                final_bullets.append("📢 **Reach:** Established track record.")
            elif max_auth_cites > 100:
                final_bullets.append("📈 **Momentum:** Authors have prior traction.")
            else:
                final_bullets.append("🌱 **Emerging:** Newer authors; relies on merit.")
        elif fame_label == "too_new":
            final_bullets.append("🆕 **Too new for impact score:** Citation data unavailable. Ranked by relevance only.")
            if p.semantic_reason:
                final_bullets.append(f"✨ **Relevance Insight:** Ranked highly because: {p.semantic_reason}")
        else:
            final_bullets.append("🌱 **Emerging:** Unknown authors.")
            
        if len(content_bullets) >= 1:
            final_bullets.append(f"🎯 **Market Fit:** {content_bullets[0]}")
        if len(content_bullets) >= 2:
            final_bullets.append(f"💡 **Contribution:** {content_bullets[1]}")

        p.prediction_explanations = final_bullets
        
        progress_bar.progress((i + 1) / len(target_papers))
        
    progress_bar.empty()
    return target_papers

def compute_keyword_match_score(paper: Paper, query_brief: str) -> float:
    """
    Lightweight keyword relevance score in [0, 1].
    Uses overlap between meaningful query terms and the paper title/abstract.
    """
    text = f"{paper.title} {paper.abstract}".lower()
    query = query_brief.lower()

    stopwords = {
        "the", "and", "or", "for", "with", "that", "this", "from", "into", "about",
        "what", "are", "your", "their", "they", "them", "using", "used", "use",
        "main", "whose", "where", "when", "which", "have", "has", "had", "been",
        "being", "also", "only", "just", "more", "most", "very", "much", "some",
        "such", "than", "then", "over", "under", "not", "without", "within",
        "papers", "paper", "interested", "looking", "brief", "contribution"
    }

    query_terms = re.findall(r"\b[a-zA-Z][a-zA-Z\-]+\b", query)
    query_terms = [t for t in query_terms if len(t) > 2 and t not in stopwords]

    if not query_terms:
        return 0.0

    matched = sum(1 for term in set(query_terms) if term in text)
    return min(matched / max(len(set(query_terms)), 1), 1.0)


def compute_recency_score(submitted_date: datetime, max_days: int = 30) -> float:
    """
    Returns a recency score in [0, 1], where newer papers score higher.
    """
    try:
        days_old = max((datetime.now().date() - submitted_date.date()).days, 0)
    except Exception:
        return 0.0

    if days_old >= max_days:
        return 0.0

    return 1.0 - (days_old / max_days)

def assign_heuristic_citations_free(papers: List[Paper], query_brief: str) -> List[Paper]:
    if not papers:
        return papers

    raw_scores = []

    for p in papers:
        semantic_score = p.semantic_relevance or 0.0
        keyword_score = compute_keyword_match_score(p, query_brief)
        recency_score = compute_recency_score(p.submitted_date)

        # Minimal multi-signal scoring
        final_score = (
            0.60 * semantic_score +
            0.25 * keyword_score +
            0.15 * recency_score
        )

        raw_scores.append(final_score)

        # Optional lightweight explanation for free mode
        reasons = []
        if keyword_score >= 0.3:
            reasons.append("strong keyword overlap with the research brief")
        if semantic_score >= 0.3:
            reasons.append("high semantic similarity to the query")
        if recency_score >= 0.8:
            reasons.append("very recent submission")

        if reasons:
            p.semantic_reason = "Matched due to " + ", ".join(reasons) + "."
        elif p.semantic_reason is None:
            p.semantic_reason = "Matched using a hybrid free-mode heuristic."

    min_s, max_s = min(raw_scores), max(raw_scores)

    for p, s in zip(papers, raw_scores):
        norm = (s - min_s) / (max_s - min_s) if max_s > min_s else 0.5
        p.predicted_citations = float(int(10 + norm * 40))

    return papers


def summarize_paper_plain_english(paper: Paper, llm_config: LLMConfig) -> str:
    prompt = textwrap.dedent(f"""
    Explain this research paper to a non-expert.
    Title: {paper.title}
    Abstract: {paper.abstract}
    
    Provide 3-6 plain English bullet points covering main idea, problem solved, and takeaways.
    """).strip()
    return call_llm(prompt, llm_config, label="plain_english_summary")


# =========================
# Streamlit UI
# =========================

PIPELINE_DESCRIPTION_MD = """
#### 1. Describe what you want

You write a short research brief in natural language about the kind of work you care about, and optionally what you are not interested in. If you leave both fields empty, the agent switches to a global mode and just looks for the most impactful recent Computer Science papers overall.

#### 2. The agent searches a curated local corpus

Instead of fetching papers live, the agent queries a pre-built local library of 40,000+ papers harvested from arXiv and Semantic Scholar. This library is refreshed on a schedule via a pipeline and, when configured, synced from cloud storage (Cloudflare R2) on startup — so you always get fast, up-to-date results without depending on external API availability at search time.

#### 3. The agent picks candidate papers using 3-stage hybrid search

Retrieval is a three-step funnel designed to be both fast and accurate:

- **Stage 1 — Keyword Recall (BM25):** Quickly narrows the corpus to papers containing terminology relevant to your brief.
- **Stage 2 — Semantic Filter (MiniLM):** An embedding model removes keyword matches that aren't actually relevant to the *meaning* of your query, keeping the top candidates by semantic similarity.
- **Stage 3 — Precision Reranking (CrossEncoder):** A deep cross-attention model does a final comparison between your brief and each candidate, selecting the best papers to pass to the AI.

#### The agent filters by venue (Optional)

If you selected a venue filter (e.g. "NeurIPS only" or "All Journals"), the agent applies it **after** the hybrid search. This ensures the agent first identifies the most semantically relevant papers from the entire corpus, and then narrows them down to your preferred venues.

#### 4. The agent judges how relevant each paper is

- In **LLM API mode** (OpenAI, Gemini, or Groq), a model reads each candidate and labels it as primary, secondary, or off topic.
- In **free local mode**, the agent uses a simple heuristic based on the embedding similarity to mark the most relevant papers as primary and the rest as secondary.

#### 5. The agent builds a citation impact set

The agent builds a set of papers to send to the citation impact step:

- It keeps all **primary** papers.
- If there are fewer than about 20, it tops up with the strongest **secondary** papers until it reaches roughly 20, when possible.
- In global mode, all candidates are used.

#### 6. The agent computes 1-year citation impact scores

- In **LLM API mode** (OpenAI, Gemini, or Groq), a model estimates a 1-year citation impact score for each paper using author citation data from Semantic Scholar and an LLM-generated narrative, and provides short explanations.
- In **free local mode**, the agent derives a citation impact score from the relevance signals and uses that to rank papers.

These scores are heuristic impact signals and are best used for ranking within this batch, not as ground truth.

**Note on New Papers:** Papers less than 5 days old often lack citation data in Semantic Scholar. These are marked as **"Too new for impact score"** and ranked purely by their relevance to your query.

#### 7. The agent ranks, summarizes, and saves results

The agent ranks papers, always showing **primary** papers first, then secondary ones. For the top N that you choose, it shows metadata, relevance signals, and links to arXiv and the PDF. In LLM API mode it also adds plain English summaries. All artifacts and a markdown report are saved in a project folder under `~/arxiv_ai_digest_projects/project_<timestamp>`, and you can download everything as a ZIP.
"""

def _bibtex_key_from_paper(p: Paper) -> str:
    # key like: smith2026_arxiv2503.12345
    if p.authors:
        first = p.authors[0].split()[-1]
        base = re.sub(r"[^A-Za-z0-9]+", "", first).lower() or "paper"
    else:
        base = "paper"
    year = str(p.submitted_date.year) if p.submitted_date else "nd"
    arx = (p.arxiv_id or "").split("v")[0].replace(".", "")
    suffix = f"_arxiv{arx}" if arx else ""
    return f"{base}{year}{suffix}"

def generate_bibtex_from_paper(p: Paper) -> str:
    key = _bibtex_key_from_paper(p)
    authors = " and ".join(p.authors) if p.authors else ""
    title = (p.title or "").replace("{", "").replace("}", "")
    year = str(p.submitted_date.year) if p.submitted_date else ""
    url = p.arxiv_url or p.pdf_url or ""
    return (
        f"@article{{{key},\n"
        f"  title={{ {title} }},\n"
        f"  author={{ {authors} }},\n"
        f"  year={{ {year} }},\n"
        f"  url={{ {url} }}\n"
        f"}}\n"
    )


def main():
    st.set_page_config(
        page_title="Research Agent",
        layout="wide",
    )

    # Startup sync from Cloudflare R2
    download_corpus_artifacts()

    # ===== FOOTER (injected early so no early return can skip it) =====
    # Uses CSS to push itself to the bottom of the viewport when content is short,
    # and sits naturally at the end of content when the page is long.
    st.markdown(
        """
        <style>
        /* Make the main Streamlit container fill at least the full viewport height */
        .stMainBlockContainer {
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }
        /* Push footer to the bottom by giving the content area above it flex-grow */
        .block-container {
            flex-grow: 1;
        }
        .page-footer {
            text-align: center;
            color: gray;
            font-size: 0.85rem;
            padding: 2rem 0 1rem 0;
            border-top: none;
            margin-top: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # We define a function to call at every exit point
    def render_footer():
        st.markdown(
            """
            <div class="page-footer">
                &copy; The Benevolent Bandwidth Foundation, Inc. &middot; Massachusetts Nonprofit Corporation. All rights reserved.<br>
                Built with ❤️ for humanity
            </div>
            """,
            unsafe_allow_html=True,
        )
    # ===== END FOOTER SETUP =====

    _main_body()
    render_footer()


def _main_body():
    st.title("🔎 Research Agent")

    
    st.write(
        """Too many important papers get lost in the noise. Most researchers and practitioners cannot reliably scan what is new recently in their area, find truly promising work, and trust that they did not miss something big."""
        " This agent helps with this problem by finding, ranking, and explaining recent AI papers on arxiv.org."
        " Run time can be lengthy if you select a large time window or a large backend LLM. Patience is a virtue for good things to come!"
    )

    # Sidebar
    with st.sidebar:
        st.header("🧠 Research Brief")

        research_brief_default = (
            "I am interested in papers whose MAIN contribution is about recommendation systems: "
            "for example, new model architectures, training strategies, evaluation methods, user or item modeling, "
            "or personalization techniques for recommenders.\n\n"
            "I especially care about work where recommendation is the central focus, not just a side example."
        )

        research_brief = st.text_area(
            "What kinds of topics are you looking for?",
            value=research_brief_default,
            height=200,
            help="Describe your research interest in natural language. Focus on what the main contribution "
                 "of the papers should be. If you leave this and the next box empty, the agent will perform "
                 "a global digest of recent Computer Science papers."
        )

        not_looking_for = st.text_area(
            "What are you NOT looking for? (optional)",
            value="Generic LLM papers that only list recommendation as one of many downstream tasks, "
                  "or papers that focus purely on language modeling, math reasoning, or scaling without "
                  "a clear recommendation specific contribution.",
            height=120,
        )

        #----Category selection UI-----
        st.markdown("### 🧩 arXiv Category")

        main_cat = st.selectbox(
            "Main category",
            list(ARXIV_CATEGORIES.keys()),
            index=0  # default to "Computer Science"
        )

        if main_cat == "All":
            # optional: allow multiselect of mains, but simplest is "All categories"
            subcats = []
            st.caption("Using all available categories.")
        else:
            subcats = st.multiselect(
                "Subcategory (choose one or more)",
                options=ARXIV_CATEGORIES[main_cat],
                default=["cs.AI", "cs.LG", "cs.HC"] if main_cat == "Computer Science" else [],
                format_func=lambda x: f"{ARXIV_CODE_TO_NAME.get(x, x)} ({x})",
                help="If you choose none, we'll use ALL subcategories from the selected main category."
            )
        # Build query string to pass into fetch (currently unused by SQL, available for expansions)
        arxiv_query = build_arxiv_category_query(main_cat, subcats)

        # --- Venue Filtering UI ---

        st.markdown("### 🏷 Venue Filter")

        venue_filter_type = st.selectbox(
            "Filter by venue",
            ["None", "All Conferences", "All Journals", "Specific Venue"],
            index=0
        )

        selected_category = None
        selected_venues = []

        if venue_filter_type == "Specific Venue":

            # First dropdown: choose category
            selected_category = st.selectbox(
                "Select type:",
                ["Conference", "Journal"]
            )

            # Depending on category show MULTISELECT
            if selected_category == "Conference":
                options = sorted(CONFERENCE_KEYWORDS)
            else:
                options = sorted(JOURNAL_KEYWORDS)

            selected_venues = st.multiselect(
                f"Select {selected_category.lower()}(s):",
                options=options
            )

        

        date_option = st.selectbox("Date Range", ["Last 3 Days", "Last Week", "Last Month"])

        st.markdown("### ⭐ Top N Highlight")
        top_n = st.slider(
            "How many top papers to highlight?",
            1, 10, 3
        )

        st.markdown("### 🔌 Provider")

        provider_label_free = "Free local model (no API key)"
        provider_label_openai = "OpenAI (API key required)"
        provider_label_gemini = "Gemini (API key required)"
        provider_label_groq = "Groq (API key required)"

        provider_choice = st.radio(
            "Choose provider",
            [provider_label_free, provider_label_openai, provider_label_gemini, provider_label_groq],
            index=0,
        )

        if provider_choice == provider_label_openai:
            provider = "openai"
        elif provider_choice == provider_label_gemini:
            provider = "gemini"
        elif provider_choice == provider_label_groq:
            provider = "groq"
        else:
            provider = "free_local"

        if provider == "openai":
            api_base = "https://api.openai.com/v1"
        elif provider == "gemini":
            api_base = "https://generativelanguage.googleapis.com"
        else:
            api_base = ""

        if provider == "openai":
            st.markdown("### 🤖 OpenAI Settings")
            api_key = st.text_input("OpenAI API Key", type="password")
            st.caption(
                "Your API key is used only in memory for this session, is never written to disk, "
                "and is never shared with anyone or any service other than OpenAI's API. "
                "When your session ends, the key is cleared from the app's state."
            )

            openai_models = [
                "gpt-5.2",
                "gpt-5",
                "gpt-5-mini",
                "gpt-5-nano",
                "gpt-4.1-mini",
                "gpt-4.1",
                "gpt-4o-mini",
                "gpt-4o",
                "o1",
            ]
            model_choice = st.selectbox(
                "OpenAI Chat model (for classification & citation impact scoring)",
                openai_models,
                index=0,
            )
            if model_choice == "Custom":
                model_name = st.text_input(
                    "Custom OpenAI Chat model name",
                    value="gpt-4.1-mini",
                    help="Example: gpt-4.1, gpt-4.1-mini, gpt-4o, gpt-4o-mini, o1, etc."
                )
            else:
                model_name = model_choice

            embedding_model_name = OPENAI_EMBEDDING_MODEL_NAME
            st.caption(f"Embeddings (OpenAI): `{embedding_model_name}`")

        elif provider == "gemini":
            st.markdown("### 🌌 Gemini Settings")
            api_key = st.text_input("Gemini API Key", type="password")
            st.caption(
                "Use an API key from Google AI Studio or Vertex AI for the Gemini API. "
                "The key is kept only in memory for this session and is never written to disk."
            )

            # Updated Gemini models list including Gemini 3 Preview
            gemini_models = [
                "gemini-3.1-pro-preview",
                "gemini-3.1-flash-lite-preview",
                "gemini-3-flash-preview",
                "gemini-2.5-pro",
                "gemini-2.5-flash",
            ]
            gemini_choice = st.selectbox(
                "Gemini Chat model (for classification & citation impact scoring)",
                gemini_models,
                index=0,
            )
            if gemini_choice == "Custom":
                model_name = st.text_input(
                    "Custom Gemini model name",
                    value="gemini-3.1-pro-preview",
                    help="Use the model identifier shown in Google AI Studio, for example `gemini-3.1-pro-preview`."
                )
            else:
                model_name = gemini_choice

            embedding_model_name = GEMINI_EMBEDDING_MODEL_NAME
            st.caption(f"Embeddings (Gemini): `{embedding_model_name}`")

        elif provider == "groq":
            st.markdown("### ⚡ Groq Settings")
            api_key = st.text_input("Groq API Key", type="password")
            st.caption(
                "Groq API keys are free. No credit card needed. "
                "Get yours at [https://console.groq.com](https://console.groq.com)"
            )

            groq_models = [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
            ]

            model_choice = st.selectbox(
                "Groq Model",
                groq_models,
                index=0,
            )

            if model_choice == "Custom":
                model_name = st.text_input(
                    "Custom Groq Model Name",
                    value="llama-3.1-70b-versatile",
                )
            else:
                model_name = model_choice

            embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
            st.caption("Using free local MiniLM embeddings (Groq does not provide embeddings).")

        else:
            api_key = ""
            model_name = "heuristic-free-local"
            embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
            st.caption(
                f"Embeddings (local): `{embedding_model_name}`.\n"
                "Classification and citation impact scoring use simple heuristics. No API key or external calls."
            )

        run_clicked = st.button("🚀 Run Pipeline")

    if run_clicked:
        _check_corpus_freshness()
        st.session_state["hide_pipeline_description"] = True

    hide_desc = st.session_state.get("hide_pipeline_description", False)

    if hide_desc:
        with st.expander("Show full pipeline description", expanded=False):
            st.markdown(PIPELINE_DESCRIPTION_MD)
    else:
        st.markdown(PIPELINE_DESCRIPTION_MD)

    # Mode and query brief
    brief_text = research_brief.strip()
    not_text = not_looking_for.strip()

    if not brief_text and not not_text:
        mode = "global"
        query_brief = (
            "User wants to see the most impactful recent AI, ML, and HCI papers in cs.AI, cs.LG, and cs.HC, "
            "without any additional topical filter."
        )
    elif not brief_text and not_text:
        mode = "broad_not_only"
        rb_prompt = (
            "User is broadly interested in recent AI, ML, and HCI work in cs.AI, cs.LG, and cs.HC."
        )
        query_brief = build_query_brief(rb_prompt, not_looking_for)
    else:
        mode = "targeted"
        query_brief = build_query_brief(research_brief, not_looking_for)

    params = {
        "research_brief": research_brief.strip(),
        "not_looking_for": not_looking_for.strip(),
        "date_option": date_option,
        "top_n": top_n,
        "model_name": model_name,
        "provider": provider,
        "venue_filter_type": venue_filter_type,
        "selected_category": selected_category,
        "selected_venues": selected_venues,
        "main_cat": main_cat,
        "subcats": subcats,
    }

    if "last_params" not in st.session_state:
        st.session_state["last_params"] = params.copy()

    if params != st.session_state["last_params"] and not run_clicked:
        for key in [
            "current_papers",
            "candidates",
            "used_papers",
            "used_label",
            "ranked_papers",
            "topN",
            "project_folder",
            "timestamp",
            "zip_bytes",
            "config",
            "mode",
            "current_start",
            "current_end",
            "plain_summaries",
        ]:
            st.session_state.pop(key, None)
        st.session_state["last_params"] = params.copy()
        st.info("Sidebar settings changed. Click **Run Pipeline** to generate new results.")
        return

    if run_clicked:
        st.session_state["last_params"] = params.copy()

    if provider == "openai":
        if not api_key or not model_name:
            if "ranked_papers" not in st.session_state:
                st.warning("Your OpenAI API key and chat model name are required to run in OpenAI mode.")
                return
    elif provider == "gemini":
        if not api_key or not model_name:
            if "ranked_papers" not in st.session_state:
                st.warning("Your Gemini API key and model name are required to run in Gemini mode.")
                return
    elif provider == "groq":
        if not api_key or not model_name:
            if "ranked_papers" not in st.session_state:
                st.warning("Your Groq API key and model name are required to run in Groq mode.")
                return
    else:
        api_key = api_key or ""
        model_name = model_name or "heuristic-free-local"

    llm_config = LLMConfig(
        api_key=api_key or "",
        model=model_name,
        api_base=api_base,
        provider=provider,
    )

    try:
        current_start, current_end = get_date_range(date_option)
    except ValueError as e:
        st.error(str(e))
        return

    st.session_state["mode"] = mode
    st.session_state["current_start"] = current_start
    st.session_state["current_end"] = current_end

    if not run_clicked and "ranked_papers" not in st.session_state:
        st.info("Fill in your research brief and settings in the sidebar, then click **Run Pipeline**.")
        return

    # 1. Project setup
    st.subheader("1. Project Setup")

    root_base_default = os.path.expanduser("~/arxiv_ai_digest_projects")
    base_folder = ensure_folder(root_base_default)

    if run_clicked or "project_folder" not in st.session_state or "timestamp" not in st.session_state:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_folder = os.path.join(base_folder, f"project_{timestamp}")
        project_folder = ensure_folder(project_folder)
        st.session_state["project_folder"] = project_folder
        st.session_state["timestamp"] = timestamp
    else:
        project_folder = st.session_state["project_folder"]
        timestamp = st.session_state["timestamp"]

    st.write(f"Project folder: `{project_folder}`")

    config = {
        "mode": mode,
        "query_brief": query_brief,
        "research_brief": research_brief,
        "not_looking_for": not_looking_for,
        "date_option": date_option,
        "current_start": str(current_start),
        "current_end": str(current_end),
        "project_folder": project_folder,
        "created_at": datetime.now().isoformat(),
        "llm_model": model_name,
        "llm_api_base": api_base,
        "embedding_model": embedding_model_name,
        "llm_provider": (
            "OpenAI" if provider == "openai"
            else "Gemini" if provider == "gemini"
            else "Groq" if provider == "groq"
            else "FreeLocalHeuristic"
        ),
        "top_n": top_n,
        "min_for_prediction": MIN_FOR_PREDICTION,
    }
    st.session_state["config"] = config
    save_json(os.path.join(project_folder, "config.json"), config)

    # 2. Fetch current papers from local corpus
    st.subheader("2. Fetch Current Papers from Corpus")

    if run_clicked or "current_papers" not in st.session_state:
        # --- Primary path: local SQLite corpus ---
        with st.spinner("Loading papers from local SQLite corpus by date window..."):
            current_papers = fetch_papers_from_db(
                start_date=current_start,
                end_date=current_end,
                category_filter=st.session_state.get("last_params", {}).get("main_cat", None),
                subcats=st.session_state.get("last_params", {}).get("subcats", None)
            )

        if current_papers:
            st.success(f"Loaded {len(current_papers)} papers from local corpus in this date range.")
        else:
            st.info("No papers found in local corpus for this specific date range.")

        # Apply venue filtering IMMEDIATELY
        before_v = len(current_papers)
        current_papers = filter_papers_by_venue(
            current_papers,
            venue_filter_type,
            selected_category,
            selected_venues
        )
        after_v = len(current_papers)
        
        if venue_filter_type != "None":
            display_sel = ", ".join(selected_venues) if selected_venues else ""
            name_string = f" → {display_sel}" if display_sel else ""
            st.info(
                f"Venue filter `{venue_filter_type}` applied{name_string}. "
                f"Remaining: {after_v} (Filtered out {before_v - after_v})"
            )

        st.session_state["current_papers"] = current_papers
    else:
        current_papers = st.session_state["current_papers"]

    if not current_papers:
        st.warning("No papers found for this date/venue combination. Please adjust filters.")
        return

    # Apply NOT filter provider-agnostically
    if not_text:
        current_papers, removed_count = filter_papers_by_not_terms(current_papers, not_text)
        st.info(f"Excluded {removed_count} papers whose title or abstract contained NOT terms.")

    st.session_state["current_papers"] = current_papers

    save_json(
        os.path.join(project_folder, "current_papers_all.json"),
        [asdict(p) for p in current_papers],
    )

    # 3. Candidate selection
    if mode == "global":
        st.subheader("3. Candidate Selection (Most Recent Papers)")
        if run_clicked or "candidates" not in st.session_state:
            sorted_papers = sorted(
                current_papers,
                key=lambda p: p.submitted_date,
                reverse=True,
            )
            candidates = sorted_papers[:150] if len(sorted_papers) > 150 else sorted_papers
            st.session_state["candidates"] = candidates
        else:
            candidates = st.session_state["candidates"]

        st.success(
            f"{len(candidates)} most recent cs.AI, cs.LG, and cs.HC papers selected as candidates (global mode)."
        )
    else:
        st.subheader("3. Embedding Based Candidate Selection")
        if run_clicked or "candidates" not in st.session_state:
            with st.spinner("Selecting top candidate papers via embeddings..."):
                candidates = select_embedding_candidates(
                    current_papers,
                    query_brief=query_brief,
                    llm_config=llm_config if provider in ("openai", "gemini", "groq") else None,
                    embedding_model=embedding_model_name,
                    provider=provider,
                    max_candidates=150,
                )
            if not candidates:
                st.warning("Embedding stage returned no candidates. Using all fetched papers as fallback.")
                candidates = current_papers
            st.session_state["candidates"] = candidates
        else:
            candidates = st.session_state["candidates"]

        st.success(f"{len(candidates)} top candidates selected by embedding similarity for further filtering.")

    save_json(
        os.path.join(project_folder, "candidates_embedding_selected.json"),
        [asdict(p) for p in candidates],
    )

    # 4. Relevance classification
    st.subheader("4. Relevance Classification")

    if mode == "global":
        st.info(
            "Global mode: no specific research brief was provided. "
            "Skipping relevance classification and treating all candidate papers as PRIMARY."
        )
        if run_clicked or any(p.focus_label is None for p in st.session_state.get("candidates", [])):
            for p in candidates:
                p.focus_label = "primary"
                p.llm_relevance_score = None
                if p.semantic_reason is None:
                    p.semantic_reason = "Global mode: no topical filtering; treated as primary."
    else:
        if provider in ("openai", "gemini", "groq"):
            provider_label = "OpenAI" if provider == "openai" else "Gemini" if provider == "gemini" else "Groq"
            if run_clicked or any(p.focus_label is None for p in candidates):
                with st.spinner(f"Classifying candidates as PRIMARY, SECONDARY, or OFF TOPIC ({provider_label})..."):
                    candidates = classify_papers_with_llm(
                        candidates,
                        query_brief=query_brief,
                        llm_config=llm_config,
                        batch_size=15,
                    )
                st.session_state["candidates"] = candidates
                
                # --- Inserted Line ---
                st.success(f"Classifying candidates as PRIMARY, SECONDARY, or OFF TOPIC ({provider_label})... Done!")

        else:
            st.info(
                "Free local mode: using a simple heuristic based on embedding similarity instead of LLM based classification."
            )
            if run_clicked or any(p.focus_label is None for p in candidates):
                candidates = heuristic_classify_papers_free(candidates)
                st.session_state["candidates"] = candidates

    save_json(
        os.path.join(project_folder, "candidates_with_classification.json"),
        [asdict(p) for p in candidates],
    )

    # 5. Build prediction set with minimum size
    st.subheader("5. Automatically Selected Papers for Citation Impact Scoring")

    if mode == "global":
        primary_papers = [p for p in candidates]
        secondary_papers: List[Paper] = []
        used_papers = primary_papers.copy()
        used_label = "Global mode: all candidate papers treated as PRIMARY and used for citation impact scoring."
        st.success(
            f"Global mode: using {len(used_papers)} most recent cs.AI, cs.LG, and cs.HC papers for citation impact scoring."
        )
    else:
        primary_papers = [p for p in candidates if p.focus_label == "primary"]
        secondary_papers = [p for p in candidates if p.focus_label == "secondary"]

        for group in (primary_papers, secondary_papers):
            group.sort(
                key=lambda p: (
                    p.llm_relevance_score if p.llm_relevance_score is not None else 0.0,
                    p.semantic_relevance if p.semantic_relevance is not None else 0.0,
                ),
                reverse=True,
            )

        used_label = ""
        if primary_papers:
            if len(primary_papers) >= MIN_FOR_PREDICTION:
                used_papers = primary_papers.copy()
                used_label = "All PRIMARY papers (enough for citation impact scoring)"
                st.success(
                    f"{len(primary_papers)} papers classified as PRIMARY. "
                    f"Using all of them for citation impact scoring (≥ {MIN_FOR_PREDICTION})."
                )
            else:
                used_papers = primary_papers.copy()
                if secondary_papers:
                    needed = MIN_FOR_PREDICTION - len(primary_papers)
                    topups = secondary_papers[:needed]
                    used_papers.extend(topups)
                    total = len(used_papers)
                    if len(secondary_papers) >= needed:
                        used_label = f"PRIMARY + top {len(topups)} SECONDARY to reach about {MIN_FOR_PREDICTION}"
                        st.success(
                            f"{len(primary_papers)} papers classified as PRIMARY. "
                            f"Added {len(topups)} top SECONDARY papers for citation impact scoring."
                        )
                    else:
                        used_label = (
                            f"All PRIMARY + all available SECONDARY "
                            f"(only {len(secondary_papers)} secondary papers, total {total} < {MIN_FOR_PREDICTION})"
                        )
                        st.info(
                            f"{len(primary_papers)} papers classified as PRIMARY. "
                            f"Only {len(secondary_papers)} SECONDARY papers available, so you have "
                            f"{total} papers in the scoring set (below the target of {MIN_FOR_PREDICTION})."
                        )
                else:
                    used_label = "All PRIMARY papers (no SECONDARY available)"
                    st.warning(
                        f"Only {len(primary_papers)} PRIMARY papers and no SECONDARY. "
                        "Using all PRIMARY papers for citation impact scoring even though this is below the "
                        f"target of {MIN_FOR_PREDICTION}."
                    )
        elif secondary_papers:
            used_papers = secondary_papers.copy()
            used_label = "All SECONDARY papers (no PRIMARY matches found)"
            st.warning(
                "No papers were classified as PRIMARY. Using all SECONDARY matches instead. "
                "These may only partially match your brief."
            )
        else:
            st.error("No candidates were classified as PRIMARY or SECONDARY. Nothing to proceed with.")
            return

    used_papers.sort(
        key=lambda p: (
            p.llm_relevance_score if p.llm_relevance_score is not None else 0.0,
            p.semantic_relevance if p.semantic_relevance is not None else 0.0,
        ),
        reverse=True,
    )

    st.session_state["used_papers"] = used_papers
    st.session_state["used_label"] = used_label

    save_json(
        os.path.join(project_folder, "used_papers_for_prediction.json"),
        [asdict(p) for p in used_papers],
    )

    st.write(
        "These are the papers that the pipeline will use for citation impact scoring. "
        "Selection is automatic based on mode, embeddings (in targeted modes), and relevance classification."
    )
    st.write(f"**Citation impact set description:** {used_label}")
    st.write(f"**Number of papers in citation impact set:** {len(used_papers)}")

    for p in used_papers:
        with st.expander(p.title, expanded=False):
            st.write(f"**Authors:** {', '.join(p.authors) if p.authors else 'Unknown'}")
            st.write(f"**Submitted:** {p.submitted_date.date().isoformat()}")
            st.write(f"[arXiv link]({p.arxiv_url}) | [PDF link]({p.pdf_url})")
            if p.focus_label:
                st.write(f"**Focus label:** {p.focus_label}")
            rel_str = f"{p.llm_relevance_score:.2f}" if p.llm_relevance_score is not None else "N/A"
            st.write(f"**Relevance score:** {rel_str}")
            sim_str = f"{p.semantic_relevance:.3f}" if p.semantic_relevance is not None else "N/A"
            if p.semantic_reason:
                st.write("**Why this paper is (or is not) relevant to your brief:**")
                st.write(p.semantic_reason)
            st.write(f"**Embedding similarity score:** {sim_str}")
            st.write("**Abstract:**")
            st.write(p.abstract)

    selected_papers = used_papers
    save_json(
        os.path.join(project_folder, "selected_papers_for_prediction.json"),
        [asdict(p) for p in selected_papers],
    )

    # 6. Citation impact scoring
    st.subheader("6. Citation Impact Scoring")

    if provider in ("openai", "gemini", "groq"):
        st.markdown("""
**How this step works (Moneyball Edition)**

We use the **Moneyball Algorithm** to predict 1-year citation impact scores.
It combines four signals:
1. **Author Fame:** Query Semantic Scholar for author citations.
2. **Content Utility:** LLM rates abstract market fit.
3. **Hype Keywords:** Bonus for trending topics.
4. **Niche Penalties:** Penalty for small fields.

**Note on New Papers:** Papers less than 5 days old often lack citation data in Semantic Scholar. These are marked as **"Too new for impact score"** and ranked purely by their relevance to your query.
""")
    elif provider == "gemini":
        st.markdown("""
**How this step works (Gemini mode)**

For each selected paper, the agent sends the title, authors, and abstract to a Gemini model (for example `gemini-3-pro-preview`) and asks it to assign a 1-year citation impact score. The model bases this score on signals such as how trendy the topic is, how novel and substantial the abstract sounds, how broad the potential audience is, and whether the work appears to come from strong labs or well known authors.

These citation impact scores are heuristic and are best used for ranking within this batch of papers, not as ground truth. They may reflect existing academic and data biases.
        """)
    elif provider == "groq":
        st.markdown("""
**How this step works (Groq mode)**

For each selected paper, the agent sends the title, authors, and abstract to a model on Groq and asks it to assign a 1-year citation impact score. The model bases this score on signals such as how trendy the topic is, how novel and substantial the abstract sounds, how broad the potential audience is, and whether the work appears to come from strong labs or well known authors.

These citation impact scores are heuristic and are best used for ranking within this batch of papers, not as ground truth.
        """)
    else:
        st.markdown("""
**How this step works (free local mode)**

In free local mode, the agent does not call any external LLM. Instead, it uses a lightweight heuristic ranking method that combines semantic similarity, keyword relevance, and recency signals to assign a numeric citation impact proxy score within the current batch. The absolute numbers are less important than the relative ranking.

These scores are heuristic and should be used as a guide for exploration rather than as formal evaluation metrics.
        """)

    if run_clicked or "ranked_papers" not in st.session_state:
        if provider in ("openai", "gemini", "groq"):
            with st.spinner("Calling LLM API to compute citation impact scores for selected papers..."):
                papers_with_pred = predict_citations_direct(
                    target_papers=selected_papers,
                    llm_config=llm_config,
                )
        else:
            with st.spinner("Computing heuristic citation impact scores from relevance signals..."):
                papers_with_pred = assign_heuristic_citations_free(selected_papers, query_brief)

        # SEPARATE into groups by focus (Primary vs Secondary vs Others)
        primaries = [p for p in papers_with_pred if p.focus_label == "primary"]
        secondaries = [p for p in papers_with_pred if p.focus_label == "secondary"]
        others = [p for p in papers_with_pred if p.focus_label not in ("primary", "secondary")]

        # Define a helper to sort any group: Scored (High->Low) THEN Unscored (High Relevance->Low)
        def sort_group(group: List[Paper]) -> List[Paper]:
            scored = [p for p in group if p.predicted_citations is not None and p.predicted_citations >= 0]
            unscored = [p for p in group if p.predicted_citations == -1.0]
            
            # Sort Scored by predicted_citations desc
            scored.sort(key=lambda p: p.predicted_citations, reverse=True)
            
            # Sort Unscored by relevance desc
            unscored.sort(key=lambda p: (
                p.llm_relevance_score if p.llm_relevance_score is not None else 0.0,
                p.semantic_relevance if p.semantic_relevance is not None else 0.0
            ), reverse=True)
            
            return scored + unscored

        ranked_papers = sort_group(primaries) + sort_group(secondaries) + sort_group(others)
        
        st.session_state["ranked_papers"] = ranked_papers
        st.session_state["has_run_once"] = True
    else:
        ranked_papers = st.session_state["ranked_papers"]

    save_json(
        os.path.join(project_folder, "selected_papers_with_predictions.json"),
        [asdict(p) for p in ranked_papers],
    )

    # 7. All selected papers ranked
    st.subheader("7. All Selected Papers (Ranked by Citation Impact Score)")

    st.caption(
        "Primary papers are ranked first (Scored → Too New), followed by Secondary papers (Scored → Too New)."
    )

    table_rows = []
    for rank, p in enumerate(ranked_papers, start=1):
        pred_val = p.predicted_citations
        if pred_val == -1.0:
            pred_display = "Too new to rate"
        else:
            pred_display = str(int(pred_val or 0)) # Force string to avoid mixed-type error
            
        focus = p.focus_label or "unknown"
        if focus == "primary":
            focus_display = "🟢 primary"
        elif focus == "secondary":
            focus_display = "🟡 secondary"
        elif focus == "off-topic":
            focus_display = "⚪ off-topic"
        else:
            focus_display = focus
        llm_rel = float(p.llm_relevance_score or 0.0)
        emb_rel = float(p.semantic_relevance or 0.0)
        table_rows.append(
            {
                "Rank": rank,
                "Citation impact score (1y)": pred_display,
                "Focus": focus_display,
                "Relevance score": round(llm_rel, 2),
                "Embedding similarity": round(emb_rel, 3),
                "Venue":p.venue or "N/A",
                "Title": p.title,
                "arXiv": p.arxiv_url,

            }
        )

    df = pd.DataFrame(table_rows)

 
    # --- CSV Export ---
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download ranked results (CSV)",
        data=csv_bytes,
        file_name=f"ranked_papers_{timestamp}.csv",
        mime="text/csv",
    )

    st.dataframe(
        df,
        width='stretch',
        hide_index=True,
        column_config={
            "arXiv": st.column_config.LinkColumn(
                label="arXiv",
                help="Open arXiv page",
                validate="^https?://.*",
                max_chars=100,
                display_text="arXiv link"
            ),
            "Citation impact score (1y)": st.column_config.TextColumn(
                label="Citation impact score (1y)",
                help="Score or 'Too new to rate'"
            )
        }
    )

    # 8. Top N highlighted
    top_n_effective = min(top_n, len(ranked_papers))
    topN = ranked_papers[:top_n_effective]
    st.session_state["topN"] = topN

    st.subheader(f"8. Top {top_n_effective} Papers (Highlighted)")

    if "plain_summaries" not in st.session_state:
        st.session_state["plain_summaries"] = {}
    plain_summaries: Dict[str, str] = st.session_state["plain_summaries"]

    for rank, p in enumerate(topN, start=1):
        st.markdown(f"### #{rank}: {p.title}")
        
        pred_val = p.predicted_citations
        if pred_val == -1.0:
            st.write(f"**Citation impact score (1 year):** Too new to rate")
        else:
            st.write(f"**Citation impact score (1 year):** {int(pred_val or 0)}")
            
        st.write(f"**Authors:** {', '.join(p.authors) if p.authors else 'Unknown'}")
        st.markdown(f"**Venue:** {p.venue or 'N/A'}")
        st.write(f"[arXiv link]({p.arxiv_url}) | [PDF link]({p.pdf_url})")
        bibtex = generate_bibtex_from_paper(p)

        col1, col2 = st.columns([1, 3])

        with col1:
            st.download_button(
                label="Download BibTeX",
                data=bibtex,
                file_name=f"{(p.arxiv_id or f'paper_{rank}')}.bib",
                mime="text/plain",
                key=f"bib_dl_{p.arxiv_id or rank}",
            )

        with col2:
            st.text_area(
                "BibTeX (copy)",
                value=bibtex,
                height=140,
                key=f"bib_txt_{p.arxiv_id or rank}",
            )
        if provider in ("openai", "gemini", "groq"):
            paper_key = p.arxiv_id or p.title
            if paper_key in plain_summaries:
                summary = plain_summaries[paper_key]
            else:
                with st.spinner("Generating plain English summary..."):
                    summary = summarize_paper_plain_english(p, llm_config)
                plain_summaries[paper_key] = summary
                st.session_state["plain_summaries"] = plain_summaries

            st.markdown("**Plain English summary:**")
            st.write(summary)

            if p.prediction_explanations:
                st.write("**Why this citation impact score:**")
                for ex in p.prediction_explanations[:3]:
                    st.write(f"- {ex}")
                # REMOVED REDUNDANT CHECK HERE
                # The logic inside predict_citations_direct ALREADY adds the relevance insight
                # to prediction_explanations if score is -1.0.
                # So we just print the list (which we did above).

        else:
            st.markdown("**Plain English summary:** only available in OpenAI / Gemini / Groq options")
            st.markdown("**Why this citation impact score:** only available in OpenAI / Gemini / Groq options")

        if p.focus_label:
            st.write(f"**Focus label:** {p.focus_label}")
        rel_str = f"{p.llm_relevance_score:.2f}" if p.llm_relevance_score is not None else "N/A"
        st.write(f"**Relevance score:** {rel_str}")
        sim_str = f"{p.semantic_relevance:.3f}" if p.semantic_relevance is not None else "N/A"
        st.write(f"**Embedding similarity score:** {sim_str}")

        if p.semantic_reason:
            st.write("**Why this paper matches your brief:**")
            st.write(p.semantic_reason)

        st.write("**Abstract:**")
        st.write(p.abstract)
        st.markdown("---")

    # 9. Markdown report for top N
    st.subheader("9. Export Top N Report")

    report_lines = [
        f"# Top {top_n_effective} Papers (Citation Impact Scores) - {datetime.now().isoformat()}",
        "## Research Brief",
        research_brief,
        "",
        "## Not Looking For (optional)",
        not_looking_for or "(none provided)",
        "",
        f"Mode: {mode}",
        f"Date range: {current_start} to {current_end}",
        f"Provider: {'OpenAI' if provider == 'openai' else 'Gemini' if provider == 'gemini' else 'Groq' if provider == 'groq' else 'Free local heuristic'}",
        f"Chat model: {model_name}",
        f"Embedding model: {embedding_model_name}",
        "",
    ]
    for rank, p in enumerate(topN, start=1):
        report_lines.append(f"## #{rank}: {p.title}")
        
        pred_val = p.predicted_citations
        if pred_val == -1.0:
            report_lines.append(f"- Citation impact score (1 year): Too new to rate")
        else:
            report_lines.append(f"- Citation impact score (1 year): {int(pred_val or 0)}")
            
        report_lines.append(f"- Authors: {', '.join(p.authors) if p.authors else 'Unknown'}")
        report_lines.append(f"- arXiv: {p.arxiv_url}")
        report_lines.append(f"- PDF: {p.pdf_url}")
        if p.focus_label:
            report_lines.append(f"- Focus label: {p.focus_label}")
        if p.llm_relevance_score is not None:
            report_lines.append(f"- Relevance score: {p.llm_relevance_score:.2f}")
        if p.semantic_relevance is not None:
            report_lines.append(f"- Embedding similarity: {p.semantic_relevance:.3f}")
        if p.semantic_reason:
            report_lines.append(f"- Relevance explanation: {p.semantic_reason}")
        if provider in ("openai", "gemini", "groq"):
            report_lines.append("- Citation impact explanations:")
            if p.prediction_explanations:
                for ex in p.prediction_explanations[:3]:
                    report_lines.append(f"  - {ex}")
                # Removed redundant check here too

        report_lines.append("")
        report_lines.append("Abstract:")
        report_lines.append(p.abstract)
        report_lines.append("")

    report_path = os.path.join(project_folder, "topN_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # 10. ZIP download
    if run_clicked or "zip_bytes" not in st.session_state:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in [
                "config.json",
                "current_papers_all.json",
                "candidates_embedding_selected.json",
                "candidates_with_classification.json",
                "used_papers_for_prediction.json",
                "selected_papers_for_prediction.json",
                "selected_papers_with_predictions.json",
                "topN_report.md",
            ]:
                fpath = os.path.join(project_folder, fname)
                if os.path.exists(fpath):
                    zf.write(fpath, arcname=fname)
        zip_buffer.seek(0)
        st.session_state["zip_bytes"] = zip_buffer.getvalue()

    zip_bytes = st.session_state["zip_bytes"]

    st.success(f"Results saved in `{project_folder}`")
    st.write("- `current_papers_all.json`")
    st.write("- `candidates_embedding_selected.json`")
    st.write("- `candidates_with_classification.json`")
    st.write("- `used_papers_for_prediction.json`")
    st.write("- `selected_papers_for_prediction.json`")
    st.write("- `selected_papers_with_predictions.json`")
    st.write("- `topN_report.md`")

    st.download_button(
        "⬇️ Download all results as ZIP",
        data=zip_bytes,
        file_name=f"research_agent_{timestamp}.zip",
        mime="application/zip",
    )


if __name__ == "__main__":
    main()
