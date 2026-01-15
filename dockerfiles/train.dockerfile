FROM ghcr.io/astral-sh/uv:python3.11-bookworm AS base
WORKDIR /app

# Copy lock + metadata først for caching
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

# If pyproject references README.md as project metadata, it must exist in the build context
COPY README.md README.md

# Install dependencies (but not the project yet)
RUN uv sync --frozen --no-install-project

# Copy source code
COPY src src/

# Install the project itself (editable install not needed; uv handles it)
RUN uv sync --frozen

# Run training as a module (more robust than file path)
ENTRYPOINT ["uv", "run", "python", "-m", "project.train"]