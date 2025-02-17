FROM python:3.10-slim

WORKDIR /app

RUN mkdir "/app/models"
RUN mkdir "/app/data"
RUN mkdir "/app/data/filtered_data"

COPY edge_node/requirements.txt /app/edge_node/

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    iputils-ping \
    dnsutils \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r /app/edge_node/requirements.txt

COPY edge_node /app/edge_node
COPY shared /app/shared

COPY edge_node/data/input_data.csv /app/data/

ENV PYTHONPATH=/app:/app/edge_node:/app/shared

EXPOSE 8081

CMD ["uvicorn", "edge_node.app_factory:app", "--host", "0.0.0.0", "--port", "8081"]
