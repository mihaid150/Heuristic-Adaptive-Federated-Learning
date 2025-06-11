FROM python:3.10-slim

WORKDIR /app

RUN mkdir "/app/models"
RUN mkdir "/app/cache"

COPY fog_node/requirements.txt /app/fog_node/

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    iputils-ping \
    dnsutils \
    iproute2  \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r /app/fog_node/requirements.txt

COPY fog_node /app/fog_node
COPY shared /app/shared

ENV PYTHONPATH=/app:/app/fog_node:/app/shared

EXPOSE 8081

CMD ["uvicorn", "fog_node.app_factory:app", "--host", "0.0.0.0", "--port", "8081"]