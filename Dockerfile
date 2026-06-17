# --- Frontend build ---
FROM node:20-alpine AS frontend-build
WORKDIR /frontend
COPY frontend/package.json ./
RUN npm install
COPY frontend/ ./
ENV VITE_API_URL=
RUN npm run build

# --- Python API + ML ---
FROM python:3.11-slim AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY mint_atlas/ mint_atlas/
COPY scripts/ scripts/
COPY data/ data/

RUN pip install --no-cache-dir -e .

# Bake datasets and model at build time (avoids cold-start training)
RUN mint-atlas generate --mint-id athens_owl \
 && mint-atlas generate --mint-id seleucid_antioch --seed 7 \
 && mint-atlas train --epochs 40

COPY --from=frontend-build /frontend/dist /app/frontend/dist

COPY scripts/docker_entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["/entrypoint.sh"]
