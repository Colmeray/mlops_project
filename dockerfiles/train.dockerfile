FROM ghcr.io/astral-sh/uv:python3.13-bookworm AS base
WORKDIR /app

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md

RUN uv sync --frozen --no-install-project

COPY configs ./configs
COPY src src/

RUN uv sync --frozen

ENTRYPOINT ["sh", "-lc", "\
  uv run -m project.data ensure-dataset && \
  uv run -m project.data preprocess --raw-root data/raw --out-root data/preprocessed && \
  uv run -m project.train \
"]
