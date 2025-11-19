# ---------- builder stage ----------
FROM python:3.10-slim as builder
WORKDIR /build

RUN apt-get update && apt-get install -y \
    gcc g++ libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 libgl1 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==2.2.1
ENV POETRY_NO_INTERACTION=1
COPY pyproject.toml poetry.lock ./
RUN poetry build -f wheel   # creates dist/*.whl

# ---------- runtime stage ----------
FROM python:3.10-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 libgl1 \
 && rm -rf /var/lib/apt/lists/*

# copy ONLY the wheel and install it (pulls in the deps)
COPY --from=builder /build/dist/*.whl .
RUN pip install --no-cache-dir *.whl && rm *.whl

# copy your code & config
COPY bsort/ ./bsort/
COPY settings.yaml ./

RUN mkdir -p /app/data /app/runs
ENV PYTHONUNBUFFERED=1 WANDB_DIR=/app/runs
CMD ["bsort", "--help"]