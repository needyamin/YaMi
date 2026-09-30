# syntax=docker/dockerfile:1
###############################################################################
# Fontaine AI — CPU image for training and serving
#
#   docker compose up                 → chat UI and API at http://localhost:8321
#   docker compose run --rm train     → one training run (tiny model)
#
# Design notes:
#   * The chat UI is built in a Node stage and served by the Python process.
#   * CPU-only PyTorch wheels keep the image ~1 GB instead of ~7 GB CUDA.
#   * Weights, tokenizer, and prepared data are NOT baked into the image —
#     they are mounted from the host (docker-compose.yml), so a new training
#     run never requires a rebuild.
###############################################################################

FROM node:22-slim AS web

WORKDIR /src
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# CPU-only PyTorch first, so the project install reuses it instead of
# pulling the CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install the project itself (bpe extra = fast byte-level tokenizers).
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[bpe,data]"

# Configs are copied AND bind-mounted read-only in compose: baked copies make
# `docker run` work standalone, the mount keeps edits rebuild-free.
COPY configs ./configs
COPY --from=web /src/dist ./web/dist

# Unprivileged runtime user; mount points must exist for :ro bind mounts.
RUN useradd --create-home --shell /usr/sbin/nologin fontaine \
    && mkdir -p /app/datasets /app/experiments \
    && chown -R fontaine:fontaine /app
USER fontaine

EXPOSE 8321

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8321/health', timeout=3).status == 200 else 1)"

ENTRYPOINT ["fontaine"]
# Fallback for bare `docker run fontaine-ai` — compose overrides this.
CMD ["serve", "--checkpoint", "/app/experiments/current/checkpoints", \
     "--tokenizer-dir", "/app/datasets/tokenizer", \
     "--inference-config", "/app/configs/inference/default.yaml", \
     "--set", "inference.server_host=0.0.0.0"]
