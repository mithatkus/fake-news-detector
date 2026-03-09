# Fake News Detector

An end-to-end NLP pipeline for binary fake news classification. Three modelling approaches — TF-IDF Logistic Regression, CNN with pretrained Word2Vec embeddings, and fine-tuned DistilBERT — are trained and evaluated on the ISOT Fake News Dataset (44,898 articles). The two best-performing models are served through a React + FastAPI application deployed on AWS Lambda and CloudFront, with per-prediction LIME explanations.

---

## Live Demo

**https://dqd84ak78c7ti.cloudfront.net**

Paste any news article, select a model, and receive a Real/Fake prediction, a confidence score, and a LIME word-importance breakdown showing which terms drove the classification.

---

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression (TF-IDF) | 0.981 | 0.976 | 0.985 | 0.980 |
| Logistic Regression (Word2Vec) | 0.939 | 0.929 | 0.946 | 0.937 |
| CNN (Word2Vec embeddings) | 0.986 | 0.990 | 0.980 | 0.985 |
| DistilBERT (fine-tuned) | 0.9999 | 1.000 | 0.9998 | 0.9999 |

5-fold cross-validation on the TF-IDF Logistic Regression:

- Accuracy: 0.981 ± 0.002, F1: 0.980 ± 0.002

TF-IDF features alone are highly discriminative on this dataset, achieving 98.1% accuracy with sub-second inference. The CNN recovers additional signal from local sequence patterns (+0.5 pp accuracy, 25% fewer errors). DistilBERT's contextual embeddings reach near-perfect accuracy (99.99%) but require GPU training and carry significantly higher inference cost. The live app exposes both the TF-IDF and CNN models.

---

## Key Findings

**Class balance.** The dataset contains 23,481 fake and 21,417 real articles (44,898 total). The 52/48 split is close enough to balanced that accuracy is a reliable metric, and no resampling was needed.

**Article length by label.** Fake articles average 438 words versus 396 for real articles, with higher variance. The longer tail in fake news likely reflects padding and filler content common in sensationalist writing.

**Misclassification patterns.** Articles misclassified by the TF-IDF model average 272 words versus 233 for correctly classified articles. Both the LR and CNN models consistently fail on the same edge cases: articles about Central American politics, transgender policy, and Russia-related topics where vocabulary overlaps substantially between real and fake sources.

**CNN versus TF-IDF error rate.** The CNN produces 126 misclassifications (1.40% error rate) against 169 for TF-IDF LR (1.88%), a 25% reduction. The improvement comes from Conv1D filters capturing local n-gram patterns that the bag-of-words TF-IDF representation discards entirely.

---

## Architecture

```
Browser
  |
  +-- CloudFront (CDN, HTTPS)
        |
        +-- S3  (React SPA static files)
        |
        +-- API Gateway HTTP API
              |
              +-- AWS Lambda  (Docker, arm64, 3 GB, 120 s)
                    |
                    +-- FastAPI + Mangum
                          |
                          +-- S3  (model artifacts ~160 MB)
```

The backend is a FastAPI application packaged as a Docker image and deployed to Lambda via the Mangum ASGI adapter. Model artifacts (TF-IDF vectorizer, logistic regression, CNN weights, Keras tokenizer) are stored in S3 and loaded lazily on first invocation, then cached in the Lambda execution context for subsequent warm calls. The React frontend is built with Vite and Tailwind CSS, served from S3 through CloudFront.

---

## Project Structure

```
fake-news-detector/
├── app/                        Streamlit app (single-model TF-IDF, local use)
│   └── app.py
├── backend/                    FastAPI backend (Lambda deployment)
│   ├── main.py                 Route definitions and prediction handlers
│   ├── preprocessing.py        Text cleaning for TF-IDF and CNN pipelines
│   ├── model_manager.py        Lazy S3 model loader with warm-invocation cache
│   ├── lime_utils.py           LIME explanation wrappers for both models
│   ├── Dockerfile
│   ├── requirements.txt        Local development dependencies
│   └── requirements-docker.txt Lambda container dependencies
├── data/
│   ├── raw/                    True.csv and Fake.csv (not tracked in git)
│   └── processed/              cleaned_isot.csv (output of notebook 01)
├── frontend/                   React + Vite + Tailwind CSS
│   └── src/
│       ├── App.jsx             Root component, state management, API calls
│       ├── api.js              predict() wrapper around POST /predict
│       └── components/
│           ├── ArticleInput.jsx    Textarea with example pool and validation
│           ├── ModelSelector.jsx   TF-IDF / CNN radio selector
│           ├── ResultCard.jsx      Prediction result with confidence bar
│           ├── ComparisonMode.jsx  Side-by-side dual-model view
│           └── LimeVisualization.jsx  Word-weight chip display
├── infrastructure/
│   ├── deploy.sh               Full AWS provisioning and deployment script
│   └── save_cnn_tokenizer.py   Regenerates keras_tokenizer.pkl from training data
├── models/                     Trained artifacts (not tracked in git, ~160 MB total)
│   ├── tfidf_vectorizer.pkl
│   ├── logistic_regression_tfidf.pkl
│   ├── cnn_model.keras
│   └── keras_tokenizer.pkl
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb
│   ├── 02_logistic_regression.ipynb
│   ├── 03_cnn_modeling.ipynb
│   └── 04_bert_modeling.ipynb
├── requirements.txt            Notebook and Streamlit app dependencies
├── README.md
└── README_AWS.md               Full AWS deployment guide
```

---

## Local Setup

**Requirements:** Python 3.10+, Node.js 18+.

**1. Clone**

```bash
git clone https://github.com/mithatkus/fake-news-detector.git
cd fake-news-detector
```

**2. Install Python dependencies**

```bash
pip install -r requirements.txt
```

**3. Download the ISOT dataset**

Download `True.csv` and `Fake.csv` from [Kaggle](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets) and place them in `data/raw/`.

**4. Run notebooks in order**

```bash
jupyter notebook
# Run 01 → 02 → 03 → 04
```

Notebook 01 generates `data/processed/cleaned_isot.csv`. Notebooks 02–04 train the models and write artifacts to `models/`.

**5. Run the Streamlit app (TF-IDF model only)**

```bash
streamlit run app/app.py
```

**6. Run the FastAPI backend locally**

```bash
cd backend
pip install -r requirements.txt
LOCAL_MODELS_DIR=../models uvicorn main:app --reload --port 8000
```

**7. Run the React frontend locally**

```bash
cd frontend
npm install
echo "VITE_API_URL=http://localhost:8000" > .env.local
npm run dev
```

---

## AWS Deployment

See [README_AWS.md](README_AWS.md) for the full step-by-step guide. The short version:

```bash
# 1. Generate the Keras tokenizer (required once after training)
python infrastructure/save_cnn_tokenizer.py

# 2. Provision everything and deploy
cd infrastructure
bash deploy.sh
```

`deploy.sh` creates the S3 model bucket, uploads artifacts, builds and pushes the Docker image to ECR, creates the Lambda function, sets up API Gateway, builds the React app, syncs it to S3, and creates a CloudFront distribution. Total runtime is approximately 10–15 minutes.

---

## Dataset

**ISOT Fake News Dataset**, collected by the Information Security and Object Technology (ISOT) Research Lab at the University of Victoria.

- 21,417 real articles sourced from Reuters
- 23,481 fake articles sourced from outlets flagged as unreliable
- Columns: `title`, `text`, `subject`, `date`
- 9 articles discarded during cleaning (empty after preprocessing)

Download: [Kaggle — ISOT Fake News Detection Datasets](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)

`True.csv` and `Fake.csv` are not tracked in this repository. Place them in `data/raw/` before running any notebooks.

---

## References

- Ahmed, H., Traore, I., & Saad, S. (2017). Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques. *Intelligent Systems and Applications in Computer Science (ISDDC).* — ISOT dataset
- Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). Distributed Representations of Words and Phrases and their Compositionality. *NeurIPS.* — Word2Vec
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *arXiv:1910.01108.* — DistilBERT
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *KDD.* — LIME
- Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS.* — SHAP
