# Change from latest to a specific version if your requirements.txt
FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src
COPY models models
COPY requirements.txt requirements.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install uvicorn
RUN pip install fastapi
RUN pip install prometheus-client
RUN pip install . --no-deps --no-cache-dir --verbose

EXPOSE 8080
CMD exec uvicorn src.pokemon_ddpm.api:app --port 8080 --host 0.0.0.0 --workers 1
