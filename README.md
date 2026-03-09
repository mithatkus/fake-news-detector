# Fake News Detector

An end-to-end NLP pipeline for binary fake news classification. Three modelling approaches are trained and evaluated on the ISOT Fake News Dataset (44,898 articles): TF-IDF Logistic Regression, CNN with pretrained Word2Vec embeddings, and fine-tuned DistilBERT. The two best-performing models are served through a React and FastAPI application deployed on AWS Lambda and CloudFront, with per-prediction LIME explanations that show which words drove each classification.

## Live Demo

https://dqd84ak78c7ti.cloudfront.net

Paste any news article, select a model, and receive a Real/Fake prediction, a confidence score, and a LIME word-importance breakdown.

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression (TF-IDF) | 0.981 | 0.976 | 0.985 | 0.980 |
| Logistic Regression (Word2Vec) | 0.939 | 0.929 | 0.946 | 0.937 |
| CNN (Word2Vec embeddings) | 0.986 | 0.990 | 0.980 | 0.985 |
| DistilBERT (fine-tuned) | 0.9999 | 1.000 | 0.9998 | 0.9999 |

Five-fold cross-validation on the TF-IDF Logistic Regression yields accuracy 0.981 ± 0.002 and F1 0.980 ± 0.002.

TF-IDF features alone are highly discriminative on this dataset, reaching 98.1% accuracy with sub-second inference. The CNN recovers additional signal from local sequence patterns, cutting the error count by 25% relative to TF-IDF LR. DistilBERT's contextual embeddings reach 99.99% accuracy but require GPU training and carry significantly higher inference cost. The live application exposes both the TF-IDF and CNN models.

## Key Findings

The dataset contains 23,481 fake and 21,417 real articles, totalling 44,898. The 52/48 class split is close enough to balanced that accuracy is a reliable metric and no resampling was needed.

Fake articles average 438 words versus 396 for real articles, with higher variance. The longer tail in fake news likely reflects padding and filler content common in sensationalist writing.

Articles misclassified by the TF-IDF model average 272 words, compared to 233 for correctly classified articles. Both the LR and CNN models consistently fail on the same edge cases: articles about Central American politics, transgender policy, and Russia-related topics where vocabulary overlaps substantially between real and fake sources.

The CNN produces 126 misclassifications (1.40% error rate) against 169 for TF-IDF LR (1.88%), a 25% reduction. The improvement comes from Conv1D filters capturing local n-gram patterns that the bag-of-words TF-IDF representation discards entirely.

## Architecture

The React frontend is built with Vite and Tailwind CSS. Static assets are stored in an S3 bucket and served through a CloudFront distribution over HTTPS. API calls from the browser go to an API Gateway HTTP API, which proxies requests to an AWS Lambda function. The Lambda runs a FastAPI application packaged as an arm64 Docker image and adapted for Lambda's invocation model via the Mangum library. On first invocation the function downloads the four model artifacts from a separate S3 bucket (TF-IDF vectorizer, logistic regression weights, CNN weights in Keras format, and the Keras tokenizer), loads them into memory, and caches them in the execution context so subsequent warm calls skip the download entirely. The Lambda is configured with 3 GB of memory and a 120-second timeout to accommodate TensorFlow's cold-start overhead and LIME's inference loop.

## Project Structure

Source is organised into six directories.

`app/` contains the original Streamlit application (`app.py`), which runs the TF-IDF model locally and is independent of the AWS stack.

`backend/` is the FastAPI service deployed to Lambda. `main.py` defines the `/health` and `/predict` routes. `preprocessing.py` implements two text-cleaning pipelines: one with stemming for the TF-IDF model and one without for the CNN. `model_manager.py` handles lazy loading from S3 and the warm-invocation cache. `lime_utils.py` wraps the LIME explainer for both models. The directory also contains the `Dockerfile`, a `requirements.txt` for local development, and a `requirements-docker.txt` for the Lambda container, which uses a different TensorFlow wheel.

`data/` holds the raw ISOT CSV files (not tracked in git) under `data/raw/` and the preprocessed output (`cleaned_isot.csv`) under `data/processed/`.

`frontend/` is the React application. `src/App.jsx` manages global state and fires API calls. `src/api.js` is a thin wrapper around `POST /predict`. The `src/components/` directory contains `ArticleInput.jsx` (textarea with a randomised example pool and a 30-word minimum validation), `ModelSelector.jsx`, `ResultCard.jsx`, `ComparisonMode.jsx` for a side-by-side view of both models running simultaneously, and `LimeVisualization.jsx`.

`infrastructure/` contains `deploy.sh`, a shell script that provisions the full AWS stack from scratch, and `save_cnn_tokenizer.py`, which regenerates `models/keras_tokenizer.pkl` from the training data. The tokenizer was not saved during the original CNN training run and must be regenerated before the first deployment.

`notebooks/` contains four Jupyter notebooks that must be run in order: EDA and preprocessing (01), logistic regression with SHAP explainability (02), CNN training with LIME explainability (03), and DistilBERT fine-tuning (04).

`models/` holds the trained artifacts. This directory is not tracked in git due to file size.

## Local Setup

Python 3.10 or later and Node.js 18 or later are required.

**1. Clone the repository**

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

**4. Run the notebooks**

```bash
jupyter notebook
```

Run notebooks 01 through 04 in order. Notebook 01 writes `data/processed/cleaned_isot.csv`. Notebooks 02 through 04 write trained artifacts to `models/`.

**5. Launch the Streamlit app (TF-IDF model only)**

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

## AWS Deployment

See [README_AWS.md](README_AWS.md) for the full step-by-step guide. In brief, run the following from the repository root:

```bash
python infrastructure/save_cnn_tokenizer.py
cd infrastructure && bash deploy.sh
```

`save_cnn_tokenizer.py` must be run once before the first deployment to regenerate `models/keras_tokenizer.pkl`. `deploy.sh` then creates the S3 model bucket, uploads all four artifacts, builds and pushes the Docker image to ECR, provisions the Lambda function and IAM role, creates the API Gateway HTTP API, builds the React app, syncs it to S3, and creates the CloudFront distribution. Total runtime is approximately 10 to 15 minutes.

## Dataset

The ISOT Fake News Dataset was collected by the Information Security and Object Technology (ISOT) Research Lab at the University of Victoria. It contains 21,417 real news articles sourced from Reuters and 23,481 fake articles sourced from outlets flagged as unreliable, with columns for title, text, subject, and date. Nine articles were discarded during preprocessing because they became empty after cleaning.

Download from Kaggle: https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets

`True.csv` and `Fake.csv` are not tracked in this repository. Place them in `data/raw/` before running any notebooks.

## References

1. Ahmed, H., Traore, I., & Saad, S. (2017). Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques. *Intelligent Systems and Applications in Computer Science (ISDDC).* (ISOT dataset)
2. Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). Distributed Representations of Words and Phrases and their Compositionality. *NeurIPS.* (Word2Vec)
3. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *arXiv:1910.01108.* (DistilBERT)
4. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *KDD.* (LIME)
5. Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS.* (SHAP)
