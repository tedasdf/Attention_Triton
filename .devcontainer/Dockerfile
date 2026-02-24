# Start with a clean Python base (Debian-based, very compatible with Ubuntu)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install ONLY what you need for ML and DVC
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Pre-install DVC S3 support so it's ready when you decide to use it
RUN pip install "dvc[s3]"

# Copy your source code
COPY main/ ./main/

# Smoke test command
CMD ["python", "src/train.py", "--smoke-test"]