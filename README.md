# 🔍 Fake News Detector

A complete NLP pipeline comparing three approaches to automated fake news detection on the ISOT Fake News Dataset: traditional machine learning (TF-IDF + Logistic Regression), deep learning (CNN with pretrained Word2Vec embeddings), and a fine-tuned transformer (DistilBERT). The project includes thorough EDA, model explainability with SHAP and LIME, error analysis, and a deployable Streamlit web application.

---

## 🚀 Live Demo

[Streamlit App](https://your-streamlit-app-url.streamlit.app) ← *replace with deployed URL*

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression (TF-IDF) | ~0.99 | ~0.99 | ~0.99 | ~0.99 | ~0.999 |
| Logistic Regression (Word2Vec) | ~0.96 | ~0.96 | ~0.96 | ~0.96 | ~0.99 |
| CNN (Word2Vec embeddings) | ~0.98 | ~0.98 | ~0.98 | ~0.98 | ~0.998 |
| DistilBERT (fine-tuned) | ~0.995 | ~0.995 | ~0.995 | ~0.995 | ~0.9995 |

*Values are placeholders — fill in after running the notebooks.*

**Takeaway:** TF-IDF Logistic Regression achieves surprisingly strong performance, while DistilBERT's contextual embeddings push accuracy to near-perfect — but at significantly higher compute cost.

---

## 🔑 Key Findings

- **[Finding from EDA notebook]** — e.g., fake news articles tend to be significantly longer than real news on average, suggesting padding or filler content.
- **[Finding from EDA notebook]** — e.g., certain subjects (e.g., "politicsNews") are overwhelmingly real while others ("News", "politics") skew heavily fake.
- **[Finding from error analysis]** — e.g., misclassified articles are disproportionately short, suggesting the model struggles with limited context.
- **[Finding from SHAP]** — e.g., the word "reuters" is the single strongest signal for real news, reflecting the dataset's sourcing from Reuters.

---

## 📁 Project Structure

```
fake-news-detector/
├── data/
│   ├── raw/                          # Original ISOT dataset CSVs (not tracked in git)
│   │   ├── True.csv                  # Real news articles
│   │   └── Fake.csv                  # Fake news articles
│   └── processed/
│       └── cleaned_isot.csv          # Preprocessed and labeled dataset
├── models/                           # Trained model artifacts
│   ├── tfidf_vectorizer.pkl          # Fitted TF-IDF vectorizer
│   ├── logistic_regression_tfidf.pkl # LR trained on TF-IDF features
│   ├── logistic_regression_w2v.pkl   # LR trained on Word2Vec embeddings
│   ├── cnn_model.keras               # Trained Keras CNN model
│   └── distilbert_model/             # Saved HuggingFace DistilBERT model
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb   # EDA, visualizations, preprocessing pipeline
│   ├── 02_logistic_regression.ipynb     # TF-IDF and Word2Vec LR models + SHAP
│   ├── 03_cnn_modeling.ipynb            # CNN model + LIME explainability
│   └── 04_bert_modeling.ipynb           # DistilBERT fine-tuning
├── app/
│   └── app.py                        # Streamlit web application
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## 🗃️ Dataset

**ISOT Fake News Dataset** — collected by the Information Security and Object Technology (ISOT) Research Lab at the University of Victoria.

- ~21,000 real news articles (sourced from Reuters)
- ~23,000 fake news articles (sourced from flagged unreliable outlets)
- Columns: `title`, `text`, `subject`, `date`

Download: [Kaggle — ISOT Fake News Detection Dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)

> **Note:** `True.csv` and `Fake.csv` are not included in this repository due to file size. Place them in `data/raw/` before running any notebooks.

---

## ⚙️ Setup

**1. Clone the repository**
```bash
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download NLTK resources** (first time only — the notebooks handle this automatically)
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

**4. Download the ISOT dataset**

Go to [Kaggle](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets), download the dataset, and place `True.csv` and `Fake.csv` in `data/raw/`.

**5. Run notebooks in order**
```bash
jupyter notebook
# Open and run 01 → 02 → 03 → 04
```

**6. Launch the Streamlit app**
```bash
streamlit run app/app.py
```

---

## 🧠 How It Works

### Preprocessing Pipeline
Raw news articles (title + text combined) are cleaned through a multi-step pipeline: Reuters source tags are stripped, URLs/mentions/hashtags removed, text lowercased, numbers and punctuation removed, tokenized with NLTK, stopwords filtered, and tokens stemmed with PorterStemmer. The result is a compact `clean_text` column used by the traditional ML and CNN models. DistilBERT uses the original (unstemmed) text since it benefits from natural language structure.

### Modeling Approaches

| Approach | Text Representation | Why It Differs |
|---|---|---|
| TF-IDF + Logistic Regression | Sparse word/bigram frequency matrix | Fast, interpretable, strong baseline; ignores word order |
| Word2Vec + Logistic Regression | Dense 300-dim average embedding | Captures semantic similarity; loses local word patterns |
| CNN + Word2Vec | Pretrained embedding + convolutional filters | Learns local n-gram patterns via learned filters |
| DistilBERT (fine-tuned) | Contextual token embeddings | Every word's meaning depends on its context; captures nuance |

### Explainability
- **SHAP (Logistic Regression):** `shap.LinearExplainer` computes each word's contribution to the prediction. The summary plot reveals globally which terms drive fake vs. real classification; waterfall plots show individual article decisions.
- **LIME (CNN):** `LimeTextExplainer` perturbs the input text by masking words, then fits a local linear model to approximate the CNN's decision boundary — revealing which words pushed the prediction in each direction.

---

## 📚 References

- Ahmed, H., Traore, I., & Saad, S. (2017). *Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques.* ISDDC. — **ISOT dataset**
- Mikolov, T., et al. (2013). *Distributed Representations of Words and Phrases and their Compositionality.* NeurIPS. — **Word2Vec**
- Sanh, V., et al. (2019). *DistilBERT, a distilled version of BERT.* arXiv:1910.01108. — **DistilBERT**
- Ribeiro, M. T., et al. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier.* KDD. — **LIME**
- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS. — **SHAP**
