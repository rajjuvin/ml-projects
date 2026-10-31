# Container image for the readmission risk web app.
#   docker build -t readmission-app .
#   docker run --rm -p 8080:8080 readmission-app      then open http://localhost:8080
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first. This layer is rebuilt only when requirements.txt changes, so code edits build quickly.
# The pinned versions are the ones the model was trained with (see models/metadata.json "libraries").
# xgboost-cpu is the same release as xgboost (same version, same "import xgboost", same model format) without the
# 300+ MB NVIDIA multi-GPU library the standard package pulls in on Linux. For a GPU variant of the image, build with
#   docker build --build-arg XGBOOST_PACKAGE=xgboost ...
# (the app would also need a code change to ask XGBoost for a GPU; the model is set up for the CPU).
ARG XGBOOST_PACKAGE=xgboost-cpu
COPY requirements.txt ./
RUN sed "s/^xgboost==/${XGBOOST_PACKAGE}==/" requirements.txt > /tmp/requirements.txt \
    && grep -q "^${XGBOOST_PACKAGE}==" /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt

# The package, then the trained model and its metadata. The build/ folder pip leaves behind is not needed to run.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps . && rm -rf build
COPY models/readmission_model.joblib models/metadata.json ./models/

# Run as an ordinary user: the app writes nothing, so it needs no special rights.
RUN useradd --create-home --uid 10001 app
USER app

# Hosts such as Cloud Run set PORT; 8080 is the default when it is not set.
ENV READMISSION_MODEL_DIR=/app/models \
    PORT=8080
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ['PORT'])"

# --factory builds the app with create_app(); the access log stays off so no request details are logged.
CMD ["sh", "-c", "exec uvicorn --factory readmission.app:create_app --host 0.0.0.0 --port ${PORT} --no-access-log"]
