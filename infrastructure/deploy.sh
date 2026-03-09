#!/usr/bin/env bash
# =============================================================================
# deploy.sh – Full AWS deployment for the Fake News Detector
# =============================================================================
# Prerequisites:
#   1. AWS CLI configured (aws configure) with IAM user fake-news-app-dev
#   2. Docker running (Docker Desktop or colima: `colima start --arch aarch64`)
#   3. Node.js + npm installed (for building the React frontend)
#   4. models/keras_tokenizer.pkl generated (run save_cnn_tokenizer.py first)
#
# Usage:
#   cd infrastructure && bash deploy.sh
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
AWS_ACCOUNT_ID="138029549417"
AWS_REGION="us-east-1"

MODEL_BUCKET="fake-news-detector-models-${AWS_ACCOUNT_ID}"
ECR_REPO="fake-news-detector-backend"
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_URI="${ECR_URI}/${ECR_REPO}:latest"

LAMBDA_FUNCTION="fake-news-detector-api"
LAMBDA_ROLE_NAME="fake-news-detector-lambda-role"
LAMBDA_MEMORY=3008    # MB  – CNN+TF needs headroom; Lambda CPU scales with memory
LAMBDA_TIMEOUT=120    # seconds – covers CNN cold-start + LIME explanation

API_GATEWAY_NAME="fake-news-detector-api"

FRONTEND_BUCKET="fake-news-detector-frontend-${AWS_ACCOUNT_ID}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="${REPO_ROOT}/models"
BACKEND_DIR="${REPO_ROOT}/backend"
FRONTEND_DIR="${REPO_ROOT}/frontend"

log() { echo ""; echo "▶  $*"; }

# ── 0. Sanity checks ──────────────────────────────────────────────────────────
log "Checking prerequisites"
command -v aws    >/dev/null || { echo "aws CLI not found"; exit 1; }
command -v docker >/dev/null || { echo "docker not found"; exit 1; }
command -v npm    >/dev/null || { echo "npm not found"; exit 1; }

if [ ! -f "${MODELS_DIR}/keras_tokenizer.pkl" ]; then
  echo ""
  echo "ERROR: models/keras_tokenizer.pkl not found."
  echo "Run this first:"
  echo "  python infrastructure/save_cnn_tokenizer.py"
  exit 1
fi

aws sts get-caller-identity --query "Account" --output text >/dev/null

# ── 1. Create model S3 bucket ─────────────────────────────────────────────────
log "Creating model bucket: ${MODEL_BUCKET}"
aws s3api create-bucket \
  --bucket "${MODEL_BUCKET}" \
  --region "${AWS_REGION}" 2>/dev/null \
  || echo "  (bucket already exists)"

aws s3api put-bucket-versioning \
  --bucket "${MODEL_BUCKET}" \
  --versioning-configuration Status=Enabled

# Block public access on the model bucket
aws s3api put-public-access-block \
  --bucket "${MODEL_BUCKET}" \
  --public-access-block-configuration \
    "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"

# ── 2. Upload model artifacts ─────────────────────────────────────────────────
log "Uploading model artifacts to S3"
for f in \
  tfidf_vectorizer.pkl \
  logistic_regression_tfidf.pkl \
  cnn_model.keras \
  keras_tokenizer.pkl
do
  if [ -f "${MODELS_DIR}/${f}" ]; then
    echo "  Uploading ${f} …"
    aws s3 cp "${MODELS_DIR}/${f}" "s3://${MODEL_BUCKET}/${f}" \
      --storage-class STANDARD_IA
  else
    echo "  WARNING: ${MODELS_DIR}/${f} not found — skipping"
  fi
done

# ── 3. Create ECR repository ──────────────────────────────────────────────────
log "Creating ECR repository: ${ECR_REPO}"
aws ecr create-repository \
  --repository-name "${ECR_REPO}" \
  --region "${AWS_REGION}" \
  --image-scanning-configuration scanOnPush=true 2>/dev/null \
  || echo "  (repo already exists)"

# ── 4. Build & push Docker image ──────────────────────────────────────────────
log "Logging into ECR"
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${ECR_URI}"

# Build for linux/arm64 (native on Apple Silicon, runs on Lambda Graviton2)
log "Building Docker image (linux/arm64)"
docker build \
  --platform linux/arm64 \
  --tag "${ECR_REPO}:latest" \
  "${BACKEND_DIR}"

log "Pushing image to ECR"
docker tag "${ECR_REPO}:latest" "${IMAGE_URI}"
docker push "${IMAGE_URI}"

# ── 5. Create Lambda IAM role ─────────────────────────────────────────────────
log "Creating Lambda IAM execution role"

TRUST_POLICY='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

LAMBDA_ROLE_ARN=$(
  aws iam create-role \
    --role-name "${LAMBDA_ROLE_NAME}" \
    --assume-role-policy-document "${TRUST_POLICY}" \
    --query "Role.Arn" --output text 2>/dev/null
) || LAMBDA_ROLE_ARN=$(
  aws iam get-role \
    --role-name "${LAMBDA_ROLE_NAME}" \
    --query "Role.Arn" --output text
)

echo "  Role ARN: ${LAMBDA_ROLE_ARN}"

aws iam attach-role-policy \
  --role-name "${LAMBDA_ROLE_NAME}" \
  --policy-arn "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole" \
  2>/dev/null || true

# Inline S3 read policy for the model bucket
aws iam put-role-policy \
  --role-name "${LAMBDA_ROLE_NAME}" \
  --policy-name "ModelBucketReadAccess" \
  --policy-document "{
    \"Version\":\"2012-10-17\",
    \"Statement\":[{
      \"Effect\":\"Allow\",
      \"Action\":[\"s3:GetObject\"],
      \"Resource\":\"arn:aws:s3:::${MODEL_BUCKET}/*\"
    }]
  }"

echo "  Waiting 15 s for IAM role to propagate …"
sleep 15

# ── 6. Deploy Lambda function ─────────────────────────────────────────────────
log "Deploying Lambda function: ${LAMBDA_FUNCTION}"

# NOTE: AWS_REGION is reserved by Lambda and cannot be set manually.
# Lambda auto-sets it; model_manager.py uses os.environ.get("AWS_REGION","us-east-1").
ENV_VARS="Variables={MODEL_BUCKET=${MODEL_BUCKET}}"

LAMBDA_EXISTS=$(
  aws lambda get-function \
    --function-name "${LAMBDA_FUNCTION}" \
    --query "Configuration.FunctionName" --output text 2>/dev/null || true
)

if [ -n "${LAMBDA_EXISTS}" ]; then
  echo "  Updating existing function …"
  aws lambda update-function-code \
    --function-name "${LAMBDA_FUNCTION}" \
    --image-uri "${IMAGE_URI}" >/dev/null

  aws lambda wait function-updated --function-name "${LAMBDA_FUNCTION}"

  aws lambda update-function-configuration \
    --function-name "${LAMBDA_FUNCTION}" \
    --memory-size "${LAMBDA_MEMORY}" \
    --timeout "${LAMBDA_TIMEOUT}" \
    --environment "${ENV_VARS}" >/dev/null
else
  echo "  Creating new function …"
  aws lambda create-function \
    --function-name "${LAMBDA_FUNCTION}" \
    --package-type Image \
    --code "ImageUri=${IMAGE_URI}" \
    --role "${LAMBDA_ROLE_ARN}" \
    --memory-size "${LAMBDA_MEMORY}" \
    --timeout "${LAMBDA_TIMEOUT}" \
    --architectures arm64 \
    --environment "${ENV_VARS}" >/dev/null
fi

echo "  Waiting for Lambda to become active …"
aws lambda wait function-active --function-name "${LAMBDA_FUNCTION}"

LAMBDA_ARN=$(
  aws lambda get-function \
    --function-name "${LAMBDA_FUNCTION}" \
    --query "Configuration.FunctionArn" --output text
)
echo "  Lambda ARN: ${LAMBDA_ARN}"

# ── 7. Set up API Gateway (HTTP API) ──────────────────────────────────────────
log "Setting up API Gateway HTTP API: ${API_GATEWAY_NAME}"

EXISTING_API_ID=$(
  aws apigatewayv2 get-apis \
    --query "Items[?Name=='${API_GATEWAY_NAME}'].ApiId | [0]" \
    --output text 2>/dev/null || true
)

if [ "${EXISTING_API_ID}" = "None" ] || [ -z "${EXISTING_API_ID}" ]; then
  API_ID=$(
    aws apigatewayv2 create-api \
      --name "${API_GATEWAY_NAME}" \
      --protocol-type HTTP \
      --cors-configuration '{"AllowOrigins":["*"],"AllowMethods":["GET","POST","OPTIONS"],"AllowHeaders":["content-type"]}' \
      --query "ApiId" --output text
  )
  echo "  Created API Gateway: ${API_ID}"
else
  API_ID="${EXISTING_API_ID}"
  echo "  Using existing API Gateway: ${API_ID}"
fi

# Lambda proxy integration
INTEGRATION_ID=$(
  aws apigatewayv2 create-integration \
    --api-id "${API_ID}" \
    --integration-type AWS_PROXY \
    --integration-uri "${LAMBDA_ARN}" \
    --payload-format-version "2.0" \
    --query "IntegrationId" --output text 2>/dev/null
) || INTEGRATION_ID=$(
  aws apigatewayv2 get-integrations \
    --api-id "${API_ID}" \
    --query "Items[0].IntegrationId" --output text
)

# Catch-all route
aws apigatewayv2 create-route \
  --api-id "${API_ID}" \
  --route-key 'ANY /{proxy+}' \
  --target "integrations/${INTEGRATION_ID}" 2>/dev/null || true

# $default stage: no stage-name path prefix (URL = https://{id}.execute-api.../health)
# This avoids needing Mangum's api_gateway_base_path configuration.
aws apigatewayv2 create-stage \
  --api-id "${API_ID}" \
  --stage-name '$default' \
  --auto-deploy 2>/dev/null || true

# Grant API Gateway permission to invoke Lambda
aws lambda add-permission \
  --function-name "${LAMBDA_FUNCTION}" \
  --statement-id "api-gateway-invoke" \
  --action "lambda:InvokeFunction" \
  --principal "apigateway.amazonaws.com" \
  --source-arn "arn:aws:execute-api:${AWS_REGION}:${AWS_ACCOUNT_ID}:${API_ID}/*" \
  2>/dev/null || true

# $default stage has no path prefix
API_URL="https://${API_ID}.execute-api.${AWS_REGION}.amazonaws.com"
echo "  API URL: ${API_URL}"

# Wait for auto-deploy to propagate
sleep 5

# ── 8. Create frontend S3 bucket ──────────────────────────────────────────────
log "Creating frontend S3 bucket: ${FRONTEND_BUCKET}"

aws s3api create-bucket \
  --bucket "${FRONTEND_BUCKET}" \
  --region "${AWS_REGION}" 2>/dev/null \
  || echo "  (bucket already exists)"

# Allow public access for static hosting
aws s3api put-public-access-block \
  --bucket "${FRONTEND_BUCKET}" \
  --public-access-block-configuration \
    "BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false"

aws s3api put-bucket-website \
  --bucket "${FRONTEND_BUCKET}" \
  --website-configuration \
    '{"IndexDocument":{"Suffix":"index.html"},"ErrorDocument":{"Key":"index.html"}}'

aws s3api put-bucket-policy \
  --bucket "${FRONTEND_BUCKET}" \
  --policy "{
    \"Version\":\"2012-10-17\",
    \"Statement\":[{
      \"Effect\":\"Allow\",
      \"Principal\":\"*\",
      \"Action\":\"s3:GetObject\",
      \"Resource\":\"arn:aws:s3:::${FRONTEND_BUCKET}/*\"
    }]
  }"

# ── 9. Build & deploy frontend ────────────────────────────────────────────────
log "Building React frontend"
cd "${FRONTEND_DIR}"

# Write the API URL into the Vite build
VITE_API_URL="${API_URL}" npm install --silent
VITE_API_URL="${API_URL}" npm run build

log "Syncing dist/ to S3"
aws s3 sync dist/ "s3://${FRONTEND_BUCKET}/" \
  --delete \
  --cache-control "max-age=86400" \
  --exclude "index.html"

# index.html must NOT be cached (for SPA routing)
aws s3 cp dist/index.html "s3://${FRONTEND_BUCKET}/index.html" \
  --cache-control "no-cache, no-store, must-revalidate"

cd "${REPO_ROOT}/infrastructure"

# ── 10. Create CloudFront distribution ────────────────────────────────────────
log "Creating CloudFront distribution"

CF_OUTPUT=$(
  aws cloudfront create-distribution --distribution-config "{
    \"CallerReference\": \"fake-news-detector-$(date +%s)\",
    \"Comment\": \"Fake News Detector Frontend\",
    \"DefaultRootObject\": \"index.html\",
    \"Origins\": {
      \"Quantity\": 1,
      \"Items\": [{
        \"Id\": \"S3Website\",
        \"DomainName\": \"${FRONTEND_BUCKET}.s3-website-${AWS_REGION}.amazonaws.com\",
        \"CustomOriginConfig\": {
          \"HTTPPort\": 80,
          \"HTTPSPort\": 443,
          \"OriginProtocolPolicy\": \"http-only\"
        }
      }]
    },
    \"DefaultCacheBehavior\": {
      \"TargetOriginId\": \"S3Website\",
      \"ViewerProtocolPolicy\": \"redirect-to-https\",
      \"AllowedMethods\": {
        \"Quantity\": 2,
        \"Items\": [\"GET\", \"HEAD\"],
        \"CachedMethods\": {\"Quantity\": 2, \"Items\": [\"GET\", \"HEAD\"]}
      },
      \"ForwardedValues\": {
        \"QueryString\": false,
        \"Cookies\": {\"Forward\": \"none\"}
      },
      \"MinTTL\": 0,
      \"DefaultTTL\": 86400,
      \"MaxTTL\": 31536000,
      \"Compress\": true
    },
    \"CustomErrorResponses\": {
      \"Quantity\": 1,
      \"Items\": [{
        \"ErrorCode\": 404,
        \"ResponseCode\": \"200\",
        \"ResponsePagePath\": \"/index.html\"
      }]
    },
    \"PriceClass\": \"PriceClass_100\",
    \"Enabled\": true
  }" 2>/dev/null || echo '{}'
)

CF_DOMAIN=$(
  echo "${CF_OUTPUT}" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('Distribution',{}).get('DomainName','(check AWS console)'))"
)

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo "  Deployment complete!"
echo "════════════════════════════════════════════════════"
echo "  API (API Gateway):  ${API_URL}"
echo "  Frontend (S3):      http://${FRONTEND_BUCKET}.s3-website-${AWS_REGION}.amazonaws.com"
echo "  Frontend (CF):      https://${CF_DOMAIN}"
echo "════════════════════════════════════════════════════"
echo ""
echo "  CloudFront propagation takes 5–15 min."
echo "  Test the API:"
echo "    curl -s -X POST ${API_URL}/predict \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"text\":\"Reuters reports that ...\",\"model\":\"tfidf\"}' | python3 -m json.tool"
echo ""
