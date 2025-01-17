# Base image
FROM python:3.11-slim AS base

WORKDIR /app

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY data data/
COPY models models/
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir --verbose

ENV PROJECT_ROOT=/app

ENTRYPOINT ["python", "-u", "src/pokemon_ddpm/train.py"]
