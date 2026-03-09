"""
LIME explanation helpers for both models.

Both functions return a list of {"word": str, "weight": float} dicts.
Positive weight → word pushed the prediction toward Real.
Negative weight → word pushed the prediction toward Fake.
"""

import logging

import numpy as np
from lime.lime_text import LimeTextExplainer

logger = logging.getLogger(__name__)

_NUM_FEATURES = 15
_TFIDF_SAMPLES = 500   # TF-IDF is fast; use more samples for stability
_CNN_SAMPLES = 50      # CNN is slower; fewer samples keeps Lambda within timeout


def _fmt(exp_list) -> list[dict]:
    return [{"word": w, "weight": float(wt)} for w, wt in exp_list]


def explain_tfidf(clean_text: str, clf, vectorizer) -> list[dict]:
    """LIME explanation for the TF-IDF Logistic Regression model."""
    explainer = LimeTextExplainer(class_names=["Fake", "Real"])

    def predict_fn(texts):
        X = vectorizer.transform(texts)
        return clf.predict_proba(X)

    try:
        exp = explainer.explain_instance(
            clean_text,
            predict_fn,
            num_features=_NUM_FEATURES,
            num_samples=_TFIDF_SAMPLES,
        )
        return _fmt(exp.as_list())
    except Exception:
        logger.exception("LIME explanation failed for TF-IDF")
        return []


def explain_cnn(clean_text: str, model, tokenizer, max_len: int = 430) -> list[dict]:
    """LIME explanation for the CNN model."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences  # noqa: PLC0415

    explainer = LimeTextExplainer(class_names=["Fake", "Real"])

    def predict_fn(texts):
        seqs = pad_sequences(
            tokenizer.texts_to_sequences(texts),
            maxlen=max_len,
            padding="post",
            truncating="post",
        )
        preds = model.predict(seqs, verbose=0).flatten()
        return np.column_stack([1 - preds, preds])

    try:
        exp = explainer.explain_instance(
            clean_text,
            predict_fn,
            num_features=_NUM_FEATURES,
            num_samples=_CNN_SAMPLES,
        )
        return _fmt(exp.as_list())
    except Exception:
        logger.exception("LIME explanation failed for CNN")
        return []
