# FREUID Challenge 2026 — reproducible inference container.
# Build:  docker build -t freuid-infer .
# Run:    docker run --rm --gpus all --network none \
#             -v /path/to/images:/data:ro -v /path/to/out:/submissions freuid-infer
# The container performs INFERENCE ONLY and needs NO network access.
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

# libGL/glib for opencv headless image decode
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY infer.py .
COPY weights/ ./weights/

# id,label CSV -> /submissions/submission.csv ; images read-only from /data
ENTRYPOINT ["python", "infer.py", "--data", "/data", "--out", "/submissions/submission.csv"]
