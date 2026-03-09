# ── Base image: AWS-managed Lambda Python runtime ─────────────────────────────
# Python 3.12 uses Amazon Linux 2023 (glibc 2.34, GCC 11+), which supports all
# manylinux wheels and avoids the GCC ≥9.3 requirement of newer numpy/h5py.
FROM public.ecr.aws/lambda/python:3.12

# ── System dependencies ───────────────────────────────────────────────────────
# Amazon Linux 2023 uses dnf (not yum).
RUN dnf install -y gcc gcc-c++ make && dnf clean all

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements-docker.txt ${LAMBDA_TASK_ROOT}/requirements-docker.txt
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements-docker.txt

# ── Bake NLTK data into the image (avoids runtime download failures) ──────────
ENV NLTK_DATA=/opt/nltk_data
RUN python -c "\
import nltk; \
nltk.download('punkt',     download_dir='/opt/nltk_data', quiet=True); \
nltk.download('punkt_tab', download_dir='/opt/nltk_data', quiet=True); \
nltk.download('stopwords', download_dir='/opt/nltk_data', quiet=True)"

# ── Application code ──────────────────────────────────────────────────────────
COPY main.py            ${LAMBDA_TASK_ROOT}/
COPY preprocessing.py   ${LAMBDA_TASK_ROOT}/
COPY model_manager.py   ${LAMBDA_TASK_ROOT}/
COPY lime_utils.py      ${LAMBDA_TASK_ROOT}/

# ── Lambda handler ────────────────────────────────────────────────────────────
CMD ["main.handler"]
