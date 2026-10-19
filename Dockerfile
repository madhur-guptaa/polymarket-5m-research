# Polaris app image (ingest / aggregate / serve) — local dev convenience.
#
# Deliberately does NOT include the "mlx" extra (see pyproject.toml):
# MLX needs direct Apple Silicon Metal access, which a Linux container has
# no path to (Docker Desktop for Mac has no Metal passthrough) — so
# `calibrate --backend mlx` isn't something this image can ever run anyway.
# Everything the image DOES run (ingest, aggregate, serve — the numpy
# predict_proba/grid/batch path) has no such dependency.
#
# docker-compose.yml bind-mounts polaris/ over the COPY below at runtime, so
# editing code on the host takes effect on container restart — no rebuild
# needed unless pyproject.toml/uv.lock change.

FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dependency layer first, cached independently of source changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

COPY polaris ./polaris

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:${PATH}"

# No CMD/ENTRYPOINT: docker-compose.yml sets `command:` per service
# (ingest | aggregate | serve — all `python -m polaris.main <subcommand>`).
