FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git \
 && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
ENV UV_PYTHON_PREFERENCE=only-managed
RUN uv python install 3.12

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
RUN uv sync --frozen --no-install-project

COPY configs ./configs
COPY src src/
RUN uv sync --frozen

# (Din entrypoint/command her)
ENTRYPOINT ["sh","-lc","uv run -m project.data ensure-dataset && uv run -m project.data preprocess --raw-root data/raw/house_plant_species --out-root data/preprocessed && uv run -m project.train"]
