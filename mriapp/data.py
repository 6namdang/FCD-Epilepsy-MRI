"""
Data access helpers for OpenNeuro ds004199 (FCD epilepsy MRI dataset).
Uses anonymous (unsigned) S3 access to the public openneuro.org bucket
-- no AWS account or credentials needed.
"""
import os
import io

import boto3
from botocore import UNSIGNED
from botocore.client import Config
import pandas as pd
import streamlit as st

BUCKET = "openneuro.org"
DATASET = "ds004199"
DATA_DIR = "data"


def _s3_client():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


@st.cache_data(show_spinner=False)
def load_participants() -> pd.DataFrame:
    """Download participants.tsv once per session and cache it."""
    s3 = _s3_client()
    key = f"{DATASET}/participants.tsv"
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()), sep="\t")
    return df


def list_subject_files(sub_id: str):
    """List all anat files available for a subject in the bucket."""
    s3 = _s3_client()
    prefix = f"{DATASET}/{sub_id}/anat/"
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    return [obj["Key"] for obj in resp.get("Contents", [])]


def download_subject(sub_id: str, dest_dir: str = DATA_DIR) -> dict:
    """
    Download T1w, FLAIR, and FLAIR lesion ROI (if published) for one subject.
    Skips files already present on disk. Returns local paths, e.g.:
        {'t1': 'data/sub-00058/anat/..._T1w.nii.gz',
         'flair': '...', 'roi': None}
    """
    s3 = _s3_client()
    keys = list_subject_files(sub_id)
    local_dir = os.path.join(dest_dir, sub_id, "anat")
    os.makedirs(local_dir, exist_ok=True)

    paths = {"t1": None, "flair": None, "roi": None}
    for key in keys:
        fname = os.path.basename(key)
        if not fname.endswith(".nii.gz"):
            continue  # skip .json sidecars

        local_path = os.path.join(local_dir, fname)
        if not os.path.exists(local_path):
            s3.download_file(BUCKET, key, local_path)

        if "roi" in fname:
            paths["roi"] = local_path
        elif "T1w" in fname:
            paths["t1"] = local_path
        elif "FLAIR" in fname:
            paths["flair"] = local_path

    return paths