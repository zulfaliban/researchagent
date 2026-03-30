"""
data_pipeline/restore_from_r2.py
Download existing artifacts from Cloudflare R2 before an incremental build.
Used primarily in GitHub Actions (CI/CD) to avoid full rebuilds on every run.
"""
import logging
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Ensure project root is on sys.path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Load env vars for R2
load_dotenv()


def restore():
    # Credentials from environment
    access_key = os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
    endpoint = os.getenv("R2_ENDPOINT")
    bucket_name = os.getenv("R2_BUCKET")

    if not all([access_key, secret_key, endpoint, bucket_name]):
        logger.warning("R2 credentials missing. Skipping restore (will run full build).")
        return

    s3 = boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    output_dir = Path("data_pipeline")
    output_dir.mkdir(exist_ok=True)

    # Core artifacts (flat files)
    artifacts = [
        "corpus.db",
        "embeddings_minilm.npy",
        "index_minilm.faiss",
        "id_map.json",
        "build_meta.json",
    ]

    logger.info("--- Restoring artifacts from R2 (%s) ---", bucket_name)
    
    for filename in artifacts:
        local_path = output_dir / filename
        remote_key = f"corpus/{filename}"
        
        try:
            logger.info("Downloading %s...", filename)
            s3.download_file(bucket_name, remote_key, str(local_path))
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            logger.info("Restored %s (%.2f MB)", filename, size_mb)
        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                logger.warning("Artifact %s not found in R2. Continuing.", filename)
            else:
                logger.error("Failed to download %s: %s", filename, e)
        except Exception as e:
            logger.error("Error restoring %s: %s", filename, e)

    # Specialized: bm25_index directory
    # List all objects with prefix 'corpus/bm25_index/'
    try:
        prefix = "corpus/bm25_index/"
        paginator = s3.get_paginator('list_objects_v2')
        
        bm25_count = 0
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            for obj in page.get('Contents', []):
                remote_key = obj['Key']
                # Relative path inside data_pipeline
                # remote_key is 'corpus/bm25_index/tokenizer.json' -> local is 'bm25_index/tokenizer.json'
                relative_path = Path(remote_key).relative_to("corpus")
                local_path = output_dir / relative_path
                
                local_path.parent.mkdir(parents=True, exist_ok=True)
                
                s3.download_file(bucket_name, remote_key, str(local_path))
                bm25_count += 1
        
        if bm25_count > 0:
            logger.info("Restored BM25 index (%d files)", bm25_count)
    except Exception as e:
        logger.error("Failed to restore BM25 index: %s", e)

    logger.info("--- Restore complete ---")


if __name__ == "__main__":
    restore()
