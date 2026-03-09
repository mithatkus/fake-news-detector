"""
Fake News Detector – FastAPI backend.

Endpoints
---------
GET  /health   – liveness probe
POST /predict  – classify an article with LIME explanation

Deployed to AWS Lambda via the Mangum ASGI adapter.
"""

import logging

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from pydantic import BaseModel, field_validator

from lime_utils import explain_cnn, explain_tfidf
from model_manager import get_cnn_models, get_tfidf_models
from preprocessing import preprocess_cnn, preprocess_tfidf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CNN_MAX_LEN = 430

app = FastAPI(
    title="Fake News Detector API",
    description="Classifies news articles as real or fake with LIME explanations.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    # Restrict to your CloudFront domain in production via the CORS_ORIGINS env var.
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str
    model: str = "tfidf"

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        if v.lower() not in {"tfidf", "cnn"}:
            raise ValueError("model must be 'tfidf' or 'cnn'")
        return v.lower()


class WordWeight(BaseModel):
    word: str
    weight: float


class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    label: str          # "Real" | "Fake"
    confidence: float   # probability of the predicted class
    real_probability: float
    fake_probability: float
    lime_words: list[WordWeight]
    model_used: str


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text cannot be empty")

    if req.model == "tfidf":
        return _predict_tfidf(text)
    return _predict_cnn(text)


# ── Prediction helpers ────────────────────────────────────────────────────────
def _predict_tfidf(text: str) -> PredictResponse:
    vectorizer, clf = get_tfidf_models()
    clean = preprocess_tfidf(text)
    X = vectorizer.transform([clean])
    proba = clf.predict_proba(X)[0]
    fake_prob, real_prob = float(proba[0]), float(proba[1])
    is_real = real_prob >= 0.5
    lime_words = explain_tfidf(clean, clf, vectorizer)
    return PredictResponse(
        label="Real" if is_real else "Fake",
        confidence=real_prob if is_real else fake_prob,
        real_probability=real_prob,
        fake_probability=fake_prob,
        lime_words=lime_words,
        model_used="tfidf",
    )


def _predict_cnn(text: str) -> PredictResponse:
    from tensorflow.keras.preprocessing.sequence import pad_sequences  # noqa: PLC0415

    model, tokenizer = get_cnn_models()
    clean = preprocess_cnn(text)
    seq = pad_sequences(
        tokenizer.texts_to_sequences([clean]),
        maxlen=CNN_MAX_LEN,
        padding="post",
        truncating="post",
    )
    pred = float(model.predict(seq, verbose=0).flatten()[0])
    real_prob, fake_prob = pred, 1.0 - pred
    is_real = real_prob >= 0.5
    lime_words = explain_cnn(clean, model, tokenizer, CNN_MAX_LEN)
    return PredictResponse(
        label="Real" if is_real else "Fake",
        confidence=real_prob if is_real else fake_prob,
        real_probability=real_prob,
        fake_probability=fake_prob,
        lime_words=lime_words,
        model_used="cnn",
    )


# ── Lambda entry point ────────────────────────────────────────────────────────
handler = Mangum(app, lifespan="off")
