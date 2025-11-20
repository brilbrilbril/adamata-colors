# Multi-stage build for smaller image size
FROM python:3.10-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=1.7.1
ENV POETRY_HOME=/opt/poetry
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV POETRY_VIRTUALENVS_CREATE=true

RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies (without the root package yet)
RUN poetry install --no-root --only main

# Copy application code
COPY bsort/ ./bsort/

# Install the root package
RUN poetry install --only-root

# Final stage
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY --from=builder /app/bsort /app/bsort
COPY --from=builder /app/pyproject.toml /app/pyproject.toml
COPY settings.yaml /app/settings.yaml

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Create directories for data
RUN mkdir -p /app/data /app/runs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV WANDB_DIR=/app/runs

# Default command
CMD ["bsort", "--help"]