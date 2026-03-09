"""
Lazy model loader.  Downloads artifacts from S3 on first use and caches them
in module-level dicts so warm Lambda invocations skip the download entirely.

Required environment variables
-------------------------------
MODEL_BUCKET       – S3 bucket that holds the model artifacts (Lambda / remote)
AWS_REGION         – AWS region (default: us-east-1)
LOCAL_MODELS_DIR   – If set, load models from this local directory instead of S3.
                     Use this for local development:
                       LOCAL_MODELS_DIR=/path/to/models uvicorn main:app --reload

Expected artifact names (S3 keys or local filenames)
-----------------------------------------------------
tfidf_vectorizer.pkl          – scikit-learn TfidfVectorizer
logistic_regression_tfidf.pkl – scikit-learn LogisticRegression
cnn_model.keras               – Keras CNN (TF .keras format)
keras_tokenizer.pkl           – Keras Tokenizer (pickled with joblib)
"""

import io
import logging
import os
import tempfile

import boto3
import joblib

logger = logging.getLogger(__name__)

_S3_BUCKET = os.environ.get("MODEL_BUCKET", "")
_AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
_LOCAL_MODELS_DIR = os.environ.get("LOCAL_MODELS_DIR", "")

# Module-level cache – survives across warm Lambda invocations
_cache: dict = {}


# ── Loaders ───────────────────────────────────────────────────────────────────
def _load_pkl(key: str):
    """Load a joblib-pickled file from local disk or S3."""
    if _LOCAL_MODELS_DIR:
        path = os.path.join(_LOCAL_MODELS_DIR, key)
        logger.info("Loading local file: %s", path)
        return joblib.load(path)
    logger.info("Downloading s3://%s/%s", _S3_BUCKET, key)
    resp = boto3.client("s3", region_name=_AWS_REGION).get_object(
        Bucket=_S3_BUCKET, Key=key
    )
    return joblib.load(io.BytesIO(resp["Body"].read()))


def _keras_model_path(key: str) -> tuple[str, bool]:
    """
    Return (path, is_temp).
    Local mode: returns the real path on disk (is_temp=False).
    S3 mode: downloads to a temp file and returns that path (is_temp=True).
    """
    if _LOCAL_MODELS_DIR:
        return os.path.join(_LOCAL_MODELS_DIR, key), False

    logger.info("Downloading s3://%s/%s", _S3_BUCKET, key)
    resp = boto3.client("s3", region_name=_AWS_REGION).get_object(
        Bucket=_S3_BUCKET, Key=key
    )
    model_bytes = resp["Body"].read()
    tmp = tempfile.NamedTemporaryFile(suffix=".keras", delete=False)
    tmp.write(model_bytes)
    tmp.close()
    return tmp.name, True


# ── Public loaders ────────────────────────────────────────────────────────────
def get_tfidf_models():
    """Return (vectorizer, clf) – cached after first load."""
    if "tfidf" not in _cache:
        vectorizer = _load_pkl("tfidf_vectorizer.pkl")
        clf = _load_pkl("logistic_regression_tfidf.pkl")
        _cache["tfidf"] = (vectorizer, clf)
        logger.info("TF-IDF models loaded")
    return _cache["tfidf"]


def get_cnn_models():
    """Return (keras_model, keras_tokenizer) – cached after first load."""
    if "cnn" not in _cache:
        from tensorflow import keras  # noqa: PLC0415

        model_path, is_temp = _keras_model_path("cnn_model.keras")
        try:
            model = keras.models.load_model(model_path)
        finally:
            if is_temp:
                os.unlink(model_path)

        tokenizer = _load_pkl("keras_tokenizer.pkl")
        _cache["cnn"] = (model, tokenizer)
        logger.info("CNN model + tokenizer loaded")
    return _cache["cnn"]
