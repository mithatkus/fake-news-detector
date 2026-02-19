import os
import re
import string

import joblib
import nltk
import numpy as np
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# ── NLTK downloads ────────────────────────────────────────────────────────────
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
)

# ── Preprocessing ─────────────────────────────────────────────────────────────
STOP_WORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()


def preprocess(text: str) -> str:
    # Remove Reuters tags
    text = re.sub(r"[A-Z\s]+\(Reuters\)\s*-\s*", "", text)
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    # Remove mentions and hashtags
    text = re.sub(r"[@#]\w+", "", text)
    # Remove "featured image via ..." patterns
    text = re.sub(r"featured image via.*", "", text, flags=re.IGNORECASE)
    # Lowercase
    text = text.lower()
    # Remove numbers and punctuation
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and stem
    tokens = [stemmer.stem(t) for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


# ── Model loading (cached) ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    clf = joblib.load(os.path.join(MODELS_DIR, "logistic_regression_tfidf.pkl"))
    return vectorizer, clf


# ── LIME helper ────────────────────────────────────────────────────────────────
def lime_highlight(text: str, clf, vectorizer):
    try:
        from lime.lime_text import LimeTextExplainer

        explainer = LimeTextExplainer(class_names=["Fake", "Real"])

        def predict_fn(texts):
            X = vectorizer.transform(texts)
            return clf.predict_proba(X)

        clean = preprocess(text)
        exp = explainer.explain_instance(clean, predict_fn, num_features=15, num_samples=500)
        return exp.as_list()
    except Exception:
        return []


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.write(
        "This tool uses machine learning to classify news articles as **real** or **fake**. "
        "It was built as a portfolio project demonstrating NLP pipelines from TF-IDF bag-of-words "
        "all the way to fine-tuned transformers (DistilBERT)."
    )

    st.header("📊 Model Performance")
    st.table(
        {
            "Model": ["LR (TF-IDF)", "LR (Word2Vec)", "CNN", "DistilBERT"],
            "Accuracy": ["~99%", "~96%", "~98%", "~99.5%"],
            "F1": ["~0.99", "~0.96", "~0.98", "~0.995"],
        }
    )

    st.header("🔗 Links")
    st.markdown("[GitHub Repository](https://github.com/your-username/fake-news-detector)")

    st.header("🗃️ Dataset")
    st.markdown(
        "**ISOT Fake News Dataset** — ~44,000 articles labeled real/fake, "
        "collected by the University of Victoria.\n\n"
        "[Download on Kaggle](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)\n\n"
        "*Ahmed, Traore & Saad (2017)*"
    )

# ── Main ───────────────────────────────────────────────────────────────────────
st.title("🔍 Fake News Detector")
st.caption("Paste any news article to classify it as real or fake using a Logistic Regression model trained on the ISOT dataset.")

# Example articles
EXAMPLES = {
    "Example: Real News": (
        "WASHINGTON (Reuters) - The White House said on Monday it was reviewing a proposed rule "
        "that would require U.S. businesses to verify the citizenship status of workers. "
        "The administration has been examining regulations on immigration enforcement across multiple agencies."
    ),
    "Example: Fake News": (
        "BREAKING: Secret documents leaked from inside the Pentagon reveal a massive government conspiracy "
        "to hide evidence of alien contact. Anonymous sources confirm that top officials have been paid "
        "to suppress the truth for decades. Share this before it gets taken down!"
    ),
}

col_ex1, col_ex2 = st.columns(2)
with col_ex1:
    if st.button("📰 Load Real News Example"):
        st.session_state["article_text"] = EXAMPLES["Example: Real News"]
with col_ex2:
    if st.button("🚨 Load Fake News Example"):
        st.session_state["article_text"] = EXAMPLES["Example: Fake News"]

article = st.text_area(
    "Paste a news article below",
    value=st.session_state.get("article_text", ""),
    height=220,
    placeholder="Paste the full text of a news article here...",
)

analyze = st.button("🔍 Analyze Article", type="primary", use_container_width=True)

if analyze:
    if not article.strip():
        st.warning("Please paste an article before clicking Analyze.")
    else:
        with st.spinner("Analyzing..."):
            try:
                vectorizer, clf = load_model()
            except FileNotFoundError:
                st.error(
                    "Model files not found. Run `notebooks/02_logistic_regression.ipynb` first "
                    "to train and save the models to `models/`."
                )
                st.stop()

            clean = preprocess(article)
            X = vectorizer.transform([clean])
            proba = clf.predict_proba(X)[0]
            lime_words = lime_highlight(article, clf, vectorizer)

        fake_prob = proba[0]
        real_prob = proba[1]
        is_real = real_prob >= 0.5
        confidence = real_prob if is_real else fake_prob

        st.divider()

        if is_real:
            st.markdown(
                "<h2 style='color:#2ecc71; text-align:center;'>✅ REAL NEWS</h2>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<h2 style='color:#e74c3c; text-align:center;'>🚨 FAKE NEWS</h2>",
                unsafe_allow_html=True,
            )

        st.markdown(
            f"<p style='text-align:center; font-size:1.2rem;'>Confidence: <strong>{confidence:.1%}</strong></p>",
            unsafe_allow_html=True,
        )

        bar_color = "#2ecc71" if is_real else "#e74c3c"
        st.markdown(
            f"""
            <div style="background:#eee; border-radius:8px; height:18px; margin-bottom:1rem;">
              <div style="background:{bar_color}; width:{confidence*100:.1f}%; height:18px; border-radius:8px;"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_r, col_f = st.columns(2)
        col_r.metric("Real probability", f"{real_prob:.1%}")
        col_f.metric("Fake probability", f"{fake_prob:.1%}")

        # LIME explanation
        if lime_words:
            st.divider()
            st.subheader("🔎 Key Words Influencing the Prediction")
            st.caption(
                "Green words pushed the model toward **Real**; Red words pushed toward **Fake**."
            )
            html_parts = []
            for word, weight in lime_words:
                color = "#27ae60" if weight > 0 else "#c0392b"
                opacity = min(1.0, abs(weight) * 5 + 0.3)
                html_parts.append(
                    f"<span style='background-color:{color}; opacity:{opacity:.2f}; "
                    f"color:white; padding:3px 7px; margin:3px; border-radius:4px; "
                    f"display:inline-block; font-size:0.95rem;'>{word}</span>"
                )
            st.markdown(" ".join(html_parts), unsafe_allow_html=True)
        else:
            st.info("Install the `lime` package to enable word-level explanations.")
