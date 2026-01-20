FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install system dependencies required for VS Code / Antigravity server
# wget is strictly required. git and curl are highly recommended for dev.
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definitions
COPY pyproject.toml uv.lock ./

# Install dependencies using the lockfile
RUN uv sync --frozen

# Add the virtual environment to the PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy source code and data configurations
# Copy source code and data
COPY src/ ./src/
COPY data/ ./data/

# Create models directory
RUN mkdir -p models

# Train the model during build so it is ready for serving
RUN python src/train.py

EXPOSE 50001

WORKDIR /app/src
# Using python direct execution for simplicity as requested, or gunicorn. 
# Since predict.py has app.run, python src/predict.py works, but gunicorn is better for prod.
# Let's use gunicorn binding to 50001.
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:50001", "predict:app"]