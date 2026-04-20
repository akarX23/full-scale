FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY inference_server ./inference_server
COPY harness ./harness
COPY models ./models
COPY data ./data

EXPOSE 8000

ENV HOST=0.0.0.0 \
    PORT=8000 \
    SERVED_MODEL=lenet \
    MODEL_TYPE=dense \
    MODEL_DIR=/app/models \
    DATA_DIR=/app/data

CMD ["python", "-m", "inference_server"]
