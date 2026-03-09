"""
One-time utility: re-fit the Keras Tokenizer on the training split and save it.

The CNN model (cnn_model.keras) was trained with a Keras Tokenizer that was
NOT saved alongside the model.  This script reconstructs the tokenizer using
the same dataset / split / settings as notebook 03_cnn_modeling.ipynb.

Usage (run from the repo root, with your ML environment active):
    python infrastructure/save_cnn_tokenizer.py

Output:
    models/keras_tokenizer.pkl  – saved with joblib; upload to S3 before deploying.
"""

import os
import sys

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from tensorflow.keras.preprocessing.text import Tokenizer
except ImportError:
    print("TensorFlow is required.  Run:  pip install tensorflow")
    sys.exit(1)

# ── Settings (must match notebook 03) ─────────────────────────────────────────
MAX_WORDS = 35_000
RANDOM_STATE = 42
TEST_SIZE = 0.2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(REPO_ROOT, "data", "processed", "cleaned_isot.csv")
OUT_PATH = os.path.join(REPO_ROOT, "models", "keras_tokenizer.pkl")


def main():
    print(f"Loading {CSV_PATH} …")
    df = pd.read_csv(CSV_PATH)

    # The CNN was trained on the `clean_text` column (pre-processed, stemmed)
    X = df["clean_text"].values
    y = df["class"].values

    X_train, _, _, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training split size: {len(X_train):,}")

    print("Fitting tokenizer …")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)
    print(f"Vocabulary size: {vocab_size:,}")

    joblib.dump(tokenizer, OUT_PATH)
    print(f"Saved → {OUT_PATH}")
    print()
    print("Next step: upload to S3")
    print(f"  aws s3 cp {OUT_PATH} s3://fake-news-detector-models-138029549417/keras_tokenizer.pkl")


if __name__ == "__main__":
    main()
