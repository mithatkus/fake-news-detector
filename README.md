# 🔍 Fake News Detector

A complete NLP pipeline that compares three approaches to automated fake news detection on the ISOT Fake News Dataset: traditional machine learning (TF-IDF + Logistic Regression), deep learning (CNN with pretrained Word2Vec embeddings), and a fine-tuned transformer (DistilBERT). The project includes thorough EDA, model explainability with SHAP and LIME, error analysis, and a deployable Streamlit web application.

---

## 🚀 Live Demo

[**Try the Fake News Detector →**](https://fake-news-detector-mithatkus.streamlit.app)

Paste any news article into the app and get an instant prediction. The app uses 
a TF-IDF Logistic Regression model trained on 44,889 articles from the ISOT 
dataset to classify articles as real or fake with 98.1% accuracy. It also shows 
a LIME explanation highlighting which specific words in your article pushed the 
prediction toward fake or real, so the model's reasoning is very 
interpretable.

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression (TF-IDF) | 0.981 | 0.976 | 0.985 | 0.980 |
| Logistic Regression (Word2Vec) | 0.939 | 0.929 | 0.946 | 0.937 |
| CNN (Word2Vec embeddings) | 0.986 | 0.990 | 0.980 | 0.985 |
| DistilBERT (fine-tuned) | 0.9999 | 1.000 | 0.9998 | 0.9999 |

5-fold cross-validation (Logistic Regression):
- TF-IDF: Accuracy 0.981 ± 0.002, F1 0.980 ± 0.002
- Word2Vec: Accuracy 0.939 ± 0.003, F1 0.936 ± 0.003

**Takeaway:** TF-IDF Logistic Regression achieves surprisingly strong performance 
at 98.1% accuracy, suggesting word-level features alone are highly discriminative 
for this dataset. The CNN improves slightly to 98.6% by capturing local text 
patterns via convolution. DistilBERT's contextual embeddings push accuracy to 
near-perfect (99.99%), but at significantly higher compute cost. As a result, TF-IDF 
Logistic Regression is the best choice when speed and simplicity matter, and that's
why it's the only available model on the streamlit app.


---

## 🔑 Key Findings

- **Class balance:** The dataset is nearly balanced with 23,481 fake and 21,417 
  real articles (44,898 total), which makes accuracy a reliable metric.
- **Fake news articles are longer on average:** Fake articles average 438 words 
  vs 396 for real articles, with higher variance which suggest more padding or 
  filler content in fake news.
- **Misclassified articles tend to be longer and more ambiguous:** Mean length 
  of misclassified articles (272 words) exceeds correctly classified ones (233 
  words). Both the LR and CNN models consistently struggle with the same edge 
  cases, namely articles about Central American politics, transgender policy, and 
  Russia-related topics that appear in both fake and real news.
- **CNN reduces errors by 25% over TF-IDF LR:** The CNN misclassifies 126 
  articles vs 169 for TF-IDF LR (1.40% vs 1.88% error rate), which shows that 
  local sequence patterns that are captured by convolution add meaningful signal 
  beyond bag-of-words features.

---

## 📁 Project Structure

The repository is organized into four main areas. The `data/` folder contains both the raw ISOT dataset CSVs and the preprocessed output — note that the raw CSVs are not tracked in git and must be downloaded separately. The `models/` folder holds all trained model artifacts: the TF-IDF vectorizer and its logistic regression model (used by the Streamlit app), the Word2Vec logistic regression, the CNN, and the fine-tuned DistilBERT. The `notebooks/` folder contains four notebooks that must be run in order — EDA and preprocessing, logistic regression with SHAP explainability, CNN modeling with LIME explainability, and DistilBERT fine-tuning. Finally, the `app/` folder contains the Streamlit web application that loads the TF-IDF model and serves predictions through a browser interface.

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
| DistilBERT (fine-tuned) | Contextual token embeddings | Every word's meaning depends on its context, captures nuance |

### Explainability
- **SHAP (Logistic Regression):** `shap.LinearExplainer` computes each word's contribution to the prediction. The summary plot reveals globally which terms drive fake vs. real classification and the waterfall plots show individual article decisions.
- **LIME (CNN):** The explainer tests what happens when individual words are removed from the article, then uses those results to identify which words had the biggest impact on the prediction — highlighting them in green (pushed toward real) or red (pushed toward fake).

---

## 📚 References

- Ahmed, H., Traore, I., & Saad, S. (2017). *Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques.* ISDDC. — **ISOT dataset**
- Mikolov, T., et al. (2013). *Distributed Representations of Words and Phrases and their Compositionality.* NeurIPS. — **Word2Vec**
- Sanh, V., et al. (2019). *DistilBERT, a distilled version of BERT.* arXiv:1910.01108. — **DistilBERT**
- Ribeiro, M. T., et al. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier.* KDD. — **LIME**
- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS. — **SHAP**
