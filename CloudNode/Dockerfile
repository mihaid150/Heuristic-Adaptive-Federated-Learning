# Use the arm64v8 Ubuntu 24.04 image
FROM ubuntu:24.04

# Install necessary packages
RUN apt-get update && apt-get install -y \
        python3 \
        python3-pip \
        python3-venv \
        openjdk-17-jdk \
        pkg-config \
        libhdf5-dev \
        build-essential \
        curl \
        && apt-get clean

# Create a virtual environment
RUN python3 -m venv /opt/venv

# Copy the requirements.txt file into the container
COPY requirements.txt /app/

# Activate the virtual environment and install dependencies
RUN /opt/venv/bin/pip install --upgrade pip setuptools wheel \
    && /opt/venv/bin/pip cache purge \
    && /opt/venv/bin/pip install -r /app/requirements.txt

# Set both PATH and PYTHONPATH
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app/scripts"

# Suppress TensorFlow warnings by setting TF_CPP_MIN_LOG_LEVEL to 3
ENV TF_CPP_MIN_LOG_LEVEL=3

# Ensure it's persisted in interactive shells
RUN echo "export PYTHONPATH=/app/scripts" >> /etc/profile.d/pythonpath.sh

WORKDIR /app

RUN mkdir -p /app/images \
    && mkdir -p /app/cache_json \
    && mkdir -p /app/scripts \
    && mkdir -p /app/scripts/model_usage \
    && mkdir -p /app/scripts/statistics_usage

COPY CloudNode-0.0.1-SNAPSHOT.jar /app/
COPY model_usage/create_init_lstm_model.py /app/scripts/model_usage/
COPY model_usage/__init__.py /app/scripts/model_usage/
COPY model_usage/aggregate_genetic_models.py /app/scripts/model_usage/
COPY statistics_usage/cloud_statistics.py /app/scripts/statistics_usage/
COPY statistics_usage/fog_statistics.py /app/scripts/statistics_usage/
COPY statistics_usage/performance_statistics.py /app/scripts/statistics_usage/
COPY statistics_usage/traffic_statistics.py /app/scripts/statistics_usage/
COPY statistics_usage/__init__.py /app/scripts/statistics_usage/
COPY model_usage/aggregate_favg_models.py /app/scripts/model_usage/
COPY model_usage/model_utils.py /app/scripts/model_usage/

RUN chmod +x /app/scripts/model_usage/*.py \
    /app/scripts/statistics_usage/*.py

EXPOSE 8080

ENTRYPOINT ["java", "-jar", "CloudNode-0.0.1-SNAPSHOT.jar"]
