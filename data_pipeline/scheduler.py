"""
data_pipeline/scheduler.py
Daily pipeline wrapper: fetch → build index.

Cron usage (Unix/macOS):
    0 2 * * * cd /repo && python data_pipeline/scheduler.py >> logs/pipeline.log 2>&1

Windows Task Scheduler: point to this script with the project root as working directory.
"""
import argparse
import os
import json
import logging
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is on sys.path when run as a script
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

# Load environment variables early (e.g. for R2 sync credentials)
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def run_command(cmd, label):
    """Run a shell command and exit on failure."""
    logger.info("--- %s ---", label)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("%s failed (exit code %d)", label, result.returncode)
        sys.exit(result.returncode)


def push_to_r2():
    """Sync artifacts to Cloudflare R2 using rclone."""
    import os
    bucket = os.getenv("R2_BUCKET")
    remote = os.getenv("RCLONE_REMOTE", "r2")
    
    if not bucket:
        logger.info("R2_BUCKET not set. Skipping R2 sync.")
        return

    logger.info("--- R2 Push (rclone) ---")
    try:
        rclone_cmd = [
            "rclone", "sync", "data_pipeline/", f"{remote}:{bucket}/corpus/",
            "--include", "corpus.db",
            "--include", "index_minilm.faiss",
            "--include", "embeddings_minilm.npy",
            "--include", "id_map.json",
            "--include", "build_meta.json",
            "--include", "bm25_index/**",
            "--transfers", "1",           # Sequential to avoid saturating unstable connections 
            "--s3-upload-concurrency", "1", # Single chunk at a time for stability
            "--retries", "5",
            "--low-level-retries", "10",
            "--progress"
        ]
        subprocess.run(rclone_cmd, check=True)
        logger.info("R2 Push complete.")
    except FileNotFoundError:
        logger.warning("rclone not found on PATH. Skipping R2 push.")
    except subprocess.CalledProcessError as e:
        logger.error("rclone sync failed with exit code %d", e.returncode)


def run() -> None:
    """Orchestrate the data pipeline based on selected mode."""
    parser = argparse.ArgumentParser(description="ResearchAgent Pipeline Scheduler")
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["arxiv", "s2", "sync", "all"],
        default="all",
        help="Pipeline mode: 'arxiv' (scout+index), 's2' (enrich+index), 'sync' (r2 only), or 'all' (default)",
    )
    parser.add_argument("--full", action="store_true", help="Pass --full to ingestion and build scripts")
    parser.add_argument("--days", type=int, default=15, help="Days to look back for arXiv scout (default: 15)")
    args = parser.parse_args()

    # Determine what to run
    run_arxiv = False
    run_s2 = False
    run_sync = False

    if args.mode == "arxiv":
        run_arxiv = True
    elif args.mode == "s2":
        run_s2 = True
    elif args.mode == "sync":
        run_sync = True
    else: # mode == "all"
        # Automated frequency logic
        meta_path = "data_pipeline/build_meta.json"

        last_arxiv = None
        last_s2 = None

        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                    if "last_arxiv_at" in meta:
                        ts = meta["last_arxiv_at"].replace("Z", "+00:00")
                        # Handle potential doubled +00:00+00:00 from older buggy runs
                        if ts.count("+00:00") > 1:
                            ts = ts.replace("+00:00+00:00", "+00:00")
                        last_arxiv = datetime.fromisoformat(ts)
                    if "last_s2_at" in meta:
                        ts = meta["last_s2_at"].replace("Z", "+00:00")
                        if ts.count("+00:00") > 1:
                            ts = ts.replace("+00:00+00:00", "+00:00")
                        last_s2 = datetime.fromisoformat(ts)
            except Exception as e:
                logger.warning("Could not parse build_meta.json for timestamps: %s", e)

        from datetime import timezone
        now = datetime.now(timezone.utc)

        # Logic: Arxiv every 2 days
        if not last_arxiv or (now - last_arxiv) >= timedelta(days=2):
            run_arxiv = True
            logger.info("Schedule: arXiv fetch due (last run: %s)", last_arxiv)
        else:
            logger.info("Schedule: Skipping arXiv fetch (not due yet)")

        # Logic: S2 every 15 days
        if not last_s2 or (now - last_s2) >= timedelta(days=15):
            run_s2 = True
            logger.info("Schedule: S2 fetch due (last run: %s)", last_s2)
        else:
            logger.info("Schedule: Skipping S2 fetch (not due yet)")

        # Always sync if we did anything, or if it's the requested mode
        run_sync = run_arxiv or run_s2

    # Execute
    if run_arxiv:
        fetch_cmd = [sys.executable, "data_pipeline/fetch_corpus.py", "--arxiv", "--days", str(args.days)]
        if args.full: fetch_cmd.append("--full")
        run_command(fetch_cmd, "Stage 1: arXiv Scout")

        index_cmd = [sys.executable, "data_pipeline/build_index.py", "--update-arxiv"]
        if args.full: index_cmd.append("--full")
        run_command(index_cmd, "Incremental Indexing (arXiv)")

    if run_s2:
        fetch_cmd = [sys.executable, "data_pipeline/fetch_corpus.py", "--s2"]
        if args.full: fetch_cmd.append("--full")
        run_command(fetch_cmd, "Stage 2: S2 Enrichment")

        index_cmd = [sys.executable, "data_pipeline/build_index.py", "--update-s2"]
        if args.full: index_cmd.append("--full")
        run_command(index_cmd, "Incremental Indexing (S2)")

    if run_sync:
        push_to_r2()

    if not (run_arxiv or run_s2 or run_sync):
        logger.info("Nothing to do today. Pipeline complete.")


if __name__ == "__main__":
    run()
