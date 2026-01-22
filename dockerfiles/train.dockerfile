FROM ghcr.io/astral-sh/uv:python3.13-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && curl -sSL https://sdk.cloud.google.com | bash -s -- --disable-prompts \
 && /root/google-cloud-sdk/bin/gcloud --version


ENV PATH="/root/google-cloud-sdk/bin:${PATH}"

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project

COPY src src/
COPY configs configs/
COPY README.md README.md

# ---------- runtime config ----------
# Where the pipeline reads/writes data (mount /app/data from the VM)
ENV RAW_ROOT=/app/data/raw
ENV OUT_ROOT=/app/data/processed

# If set at runtime, ensure-dataset will pull from GCS
# Example: gs://plant_dataset_mlops_1234/datasets/raw/house_plant_species
ENV DATA_GCS_URI=""

# ---------- run pipeline ----------
CMD ["bash", "-lc", "\
    set -e; \
    uv run python src/project/data.py ensure-dataset --data-dir \"$RAW_ROOT\"; \
    uv run python src/project/data.py preprocess \"$RAW_ROOT\" \"$OUT_ROOT\"; \
    uv run python src/project/train.py \
  "]

