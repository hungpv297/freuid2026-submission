# FREUID Challenge 2026 — reproducible inference container.
# Build:  docker build -t freuid-infer .
# Run:    docker run --rm --gpus all --network none \
#             -v /path/to/images:/data:ro -v /path/to/out:/submissions freuid-infer
# The container performs INFERENCE ONLY and needs NO network access.
#
# Base image pinned by digest for reproducibility (pytorch 2.7.1 / CUDA 12.6).
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime@sha256:2b59b1b91885677814f78be1f8df48a25d5dc952eb6580eaecfefca510f9afd3

# Organizer-sandbox path contract (matches the FREUID starter kit).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FREUID_DATA_DIR=/data \
    FREUID_OUTPUT_DIR=/submissions \
    FREUID_SUBMISSION_PATH=/submissions/submission.csv

# System libraries for OpenCV image decode.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY infer.py .
COPY weights/ ./weights/

# infer.py defaults to /data and /submissions/submission.csv (or $FREUID_* env).
ENTRYPOINT ["python", "/app/infer.py"]
CMD []
