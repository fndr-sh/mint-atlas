# --- Frontend build ---
FROM node:20-alpine AS frontend-build
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci || npm install
COPY frontend/ ./
ENV VITE_API_URL=
RUN npm run build

# --- Python API + ML ---
FROM python:3.11-slim AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# README required by hatchling metadata (pyproject.toml readme = "README.md")
COPY pyproject.toml README.md LICENSE ./
COPY mint_atlas/ mint_atlas/
COPY scripts/ scripts/
COPY data/ data/

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir .

# Bake datasets and model at build time (avoids cold-start training)
RUN mint-atlas generate --mint-id athens_owl \
 && mint-atlas generate --mint-id seleucid_antioch --seed 7 \
 && mint-atlas train --epochs 40

COPY --from=frontend-build /frontend/dist /app/frontend/dist

COPY scripts/docker_entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

ENV PORT=7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health')"

CMD ["/entrypoint.sh"]
