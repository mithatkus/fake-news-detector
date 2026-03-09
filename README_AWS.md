# Fake News Detector – AWS Deployment Guide

This guide walks through deploying the full-stack application to AWS:

- **Backend** → AWS Lambda (Docker container) + API Gateway HTTP API
- **Frontend** → S3 static hosting + CloudFront CDN
- **Models** → S3 bucket (`fake-news-detector-models-138029549417`)

---

## Architecture

```
User → CloudFront → S3 (React SPA)
                ↓ (API calls)
     API Gateway HTTP API
                ↓
          AWS Lambda
     (FastAPI + Mangum)
                ↓
     S3 (model artifacts)
```

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| AWS CLI | ≥ 2.x | Provision resources |
| Docker Desktop | any | Build Lambda image |
| Node.js + npm | ≥ 18 | Build React app |
| Python | ≥ 3.10 | Run tokenizer script |
| TensorFlow | 2.16+ | Save CNN tokenizer |

### Configure AWS CLI

```bash
aws configure
# AWS Access Key ID:     <fake-news-app-dev key>
# AWS Secret Access Key: <secret>
# Default region:        us-east-1
# Default output format: json
```

Verify:

```bash
aws sts get-caller-identity
# Should show Account: 138029549417
```

---

## Step 1 – Save the CNN Keras Tokenizer

The CNN model (`cnn_model.keras`) requires a matching Keras `Tokenizer` that was fit during training but was not saved. Run this script once to regenerate it:

```bash
# Activate your ML environment (needs tensorflow + pandas + scikit-learn)
python infrastructure/save_cnn_tokenizer.py
```

Expected output:
```
Loading data/processed/cleaned_isot.csv …
Training split size: 35,911
Fitting tokenizer …
Vocabulary size: 35,000
Saved → models/keras_tokenizer.pkl
```

You should now have `models/keras_tokenizer.pkl` (~8 MB).

---

## Step 2 – Verify model artifacts

All four files must exist in `models/` before deploying:

```
models/
├── tfidf_vectorizer.pkl          (~50 MB)
├── logistic_regression_tfidf.pkl (~40 KB)
├── cnn_model.keras               (~60 MB)
└── keras_tokenizer.pkl           (~8 MB)   ← generated in Step 1
```

---

## Step 3 – Run deploy.sh

```bash
cd infrastructure
bash deploy.sh
```

The script performs these steps in order:

| Step | Action |
|------|--------|
| 1 | Create `fake-news-detector-models-138029549417` S3 bucket |
| 2 | Upload all four model artifacts |
| 3 | Create ECR repository |
| 4 | Build + push Docker image (`linux/amd64`) |
| 5 | Create Lambda IAM execution role with S3 read access |
| 6 | Create / update Lambda function (1024 MB, 60 s timeout) |
| 7 | Create API Gateway HTTP API with `ANY /{proxy+}` route |
| 8 | Create frontend S3 bucket with public static hosting |
| 9 | Build React app with `VITE_API_URL` and sync to S3 |
| 10 | Create CloudFront distribution |

Total runtime: ~10–15 minutes (dominated by Docker push and CloudFront propagation).

---

## Step 4 – Verify the API

```bash
API_URL=https://<your-api-id>.execute-api.us-east-1.amazonaws.com/prod

# Health check
curl "${API_URL}/health"
# {"status":"ok"}

# TF-IDF prediction
curl -s -X POST "${API_URL}/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"Reuters reports the White House reviewed new immigration rules.","model":"tfidf"}' \
  | python3 -m json.tool

# CNN prediction (takes ~5–10 s on first warm-up)
curl -s -X POST "${API_URL}/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"Reuters reports the White House reviewed new immigration rules.","model":"cnn"}' \
  | python3 -m json.tool
```

Expected response shape:
```json
{
  "label": "Real",
  "confidence": 0.985,
  "real_probability": 0.985,
  "fake_probability": 0.015,
  "lime_words": [
    {"word": "reuters", "weight": 0.124},
    {"word": "white", "weight": 0.083}
  ],
  "model_used": "tfidf"
}
```

---

## Step 5 – Verify the frontend

1. CloudFront takes **5–15 minutes** to propagate after creation.
2. Open `https://<cf-domain>.cloudfront.net` in your browser.
3. Paste an article, choose a model, and click **Analyze Article**.

The deploy script prints the CloudFront URL at the end.

---

## Environment variables

### Lambda

| Variable | Value |
|----------|-------|
| `MODEL_BUCKET` | `fake-news-detector-models-138029549417` |
| `AWS_REGION` | `us-east-1` |

Set via `--environment` in deploy.sh; update with:

```bash
aws lambda update-function-configuration \
  --function-name fake-news-detector-api \
  --environment "Variables={MODEL_BUCKET=...,AWS_REGION=us-east-1}"
```

### Frontend

| Variable | Purpose |
|----------|---------|
| `VITE_API_URL` | Injected at build time by `deploy.sh`; no runtime config needed |

---

## Re-deploying after code changes

### Backend only (no model changes)

```bash
cd infrastructure
# Rebuild and push image, then update Lambda
docker build --platform linux/amd64 -t fake-news-detector-backend:latest ../backend/
docker tag fake-news-detector-backend:latest \
  138029549417.dkr.ecr.us-east-1.amazonaws.com/fake-news-detector-backend:latest
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin \
    138029549417.dkr.ecr.us-east-1.amazonaws.com
docker push 138029549417.dkr.ecr.us-east-1.amazonaws.com/fake-news-detector-backend:latest
aws lambda update-function-code \
  --function-name fake-news-detector-api \
  --image-uri 138029549417.dkr.ecr.us-east-1.amazonaws.com/fake-news-detector-backend:latest
```

### Frontend only

```bash
cd frontend
VITE_API_URL=https://<api-id>.execute-api.us-east-1.amazonaws.com/prod npm run build
aws s3 sync dist/ s3://fake-news-detector-frontend-138029549417/ --delete
# Invalidate CloudFront cache
aws cloudfront create-invalidation \
  --distribution-id <cf-dist-id> \
  --paths "/*"
```

---

## Troubleshooting

### Lambda cold starts are slow (~30 s)

The Docker image is large (TensorFlow + models loaded from S3). For production traffic, enable **Provisioned Concurrency**:

```bash
aws lambda put-provisioned-concurrency-config \
  --function-name fake-news-detector-api \
  --qualifier "\$LATEST" \
  --provisioned-concurrent-executions 1
```

This keeps one instance warm at all times (~$15/month extra).

### `Task timed out after 60.00 seconds` on CNN

LIME with the CNN runs up to 150 predictions × 430-token sequences. Increase Lambda timeout:

```bash
aws lambda update-function-configuration \
  --function-name fake-news-detector-api \
  --timeout 90
```

Or reduce `_CNN_SAMPLES` in `backend/lime_utils.py` and redeploy.

### `keras_tokenizer.pkl` missing error

Run Step 1 again and re-upload:

```bash
python infrastructure/save_cnn_tokenizer.py
aws s3 cp models/keras_tokenizer.pkl \
  s3://fake-news-detector-models-138029549417/keras_tokenizer.pkl
```

### CORS errors in the browser

The API Gateway is configured with `AllowOrigins=*`. If you restrict it to your CloudFront domain later:

```bash
aws apigatewayv2 update-api \
  --api-id <api-id> \
  --cors-configuration "AllowOrigins=https://<cf-domain>.cloudfront.net,AllowMethods=GET POST OPTIONS,AllowHeaders=content-type"
```

### Local development

**Backend:**
```bash
cd backend
pip install -r requirements.txt
export MODEL_BUCKET=fake-news-detector-models-138029549417
export AWS_REGION=us-east-1
uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
echo "VITE_API_URL=http://localhost:8000" > .env.local
npm run dev
```

---

## Cost estimate (light traffic)

| Service | Estimated monthly cost |
|---------|----------------------|
| Lambda (1M requests × 5 s × 1 GB) | ~$25 |
| API Gateway (1M requests) | ~$1 |
| S3 (model storage ~160 MB + frontend) | ~$0.01 |
| CloudFront (10 GB transfer) | ~$1 |
| ECR (image storage) | ~$0.10 |
| **Total** | **~$27/month** |

Provisioned Concurrency adds ~$15/month per warm instance.

---

## Cleanup

```bash
# Delete Lambda
aws lambda delete-function --function-name fake-news-detector-api

# Delete API Gateway
aws apigatewayv2 delete-api --api-id <api-id>

# Empty and delete model bucket
aws s3 rm s3://fake-news-detector-models-138029549417 --recursive
aws s3api delete-bucket --bucket fake-news-detector-models-138029549417

# Empty and delete frontend bucket
aws s3 rm s3://fake-news-detector-frontend-138029549417 --recursive
aws s3api delete-bucket --bucket fake-news-detector-frontend-138029549417

# Delete ECR repo
aws ecr delete-repository \
  --repository-name fake-news-detector-backend --force

# Disable and delete CloudFront distribution (must disable first, wait ~15 min)
aws cloudfront update-distribution --id <dist-id> --if-match <etag> \
  --distribution-config file://disabled-cf-config.json
aws cloudfront delete-distribution --id <dist-id> --if-match <etag>
```
