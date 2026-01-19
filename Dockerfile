FROM python:3.12-slim-bookworm

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy dependency definitions
COPY pyproject.toml uv.lock ./

# Install dependencies system-wide
RUN uv pip install --system --require-hashes -r pyproject.toml

# Copy source code and data configurations
COPY src/ ./src/
COPY data/data.yaml ./data/

# Create models directory
RUN mkdir -p models

EXPOSE 9696

WORKDIR /app/src
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]
