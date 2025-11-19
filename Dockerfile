# ----------- builder -----------
FROM python:3.10-slim AS builder
WORKDIR /build

# system deps needed to compile
RUN apt-get update && apt-get install -y \
    gcc g++ libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 libgl1 \
 && rm -rf /var/lib/apt/lists/*

# install poetry and build the wheel
RUN pip install --no-cache-dir poetry==2.2.1
COPY pyproject.toml poetry.lock ./
RUN poetry build -f wheel

# ----------- runtime -----------
FROM python:3.10-slim
WORKDIR /app

# runtime libs only
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 libgl1 \
 && rm -rf /var/lib/apt/lists/*

# copy wheel and install (pulls only runtime deps)
COPY --from=builder /build/dist/*.whl .
RUN pip install --no-cache-dir *.whl && rm *.whl

# copy application files
COPY bsort/ ./bsort/
COPY settings.yaml ./

RUN mkdir -p /app/data /app/runs
ENV PYTHONUNBUFFERED=1 WANDB_DIR=/app/runs
CMD ["bsort", "--help"]