FROM python:3.10-slim

WORKDIR /app

COPY cloud_node/requirements.txt /app/cloud_node/

RUN mkdir "/app/models"
RUN mkdir "/app/cache"

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    iputils-ping \
    dnsutils \
    iproute2 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r /app/cloud_node/requirements.txt

COPY cloud_node /app/cloud_node
COPY shared /app/shared

ENV PYTHONPATH=/app:/app/cloud_node:/app/shared

EXPOSE 8081

CMD ["uvicorn", "cloud_node.app_factory:app", "--host", "0.0.0.0", "--port", "8081"]
