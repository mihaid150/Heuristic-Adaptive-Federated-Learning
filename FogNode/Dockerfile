FROM ubuntu:24.04

RUN apt-get update && apt-get install -y \
        python3 \
        python3-pip \
        python3-venv \
        openjdk-17-jdk \
        pkg-config \
        libhdf5-dev \
        && apt-get clean

RUN python3 -m venv /opt/venv

COPY requirements.txt /app/

# Activate the virtual environment and install dependencies
RUN /opt/venv/bin/pip install --upgrade pip setuptools wheel \
    && /opt/venv/bin/pip cache purge \
    && /opt/venv/bin/pip install -r /app/requirements.txt

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app/scripts"

# Add the following line to ensure it's persisted in interactive shells
RUN echo "export PYTHONPATH=/app/scripts" >> /etc/profile.d/pythonpath.sh

# Suppress TensorFlow warnings by setting TF_CPP_MIN_LOG_LEVEL to 3
ENV TF_CPP_MIN_LOG_LEVEL=3

WORKDIR /app

RUN mkdir -p /app/cache_json \
    && mkdir -p /app/scripts/ \
    && mkdir -p /app/scripts/model_usage

COPY FogNode-0.0.1-SNAPSHOT.jar /app/
COPY model_usage/__init__.py /app/scripts/__init__.py
COPY model_usage/aggregate_genetic_models.py /app/scripts/model_usage/
COPY model_usage/aggregate_favg_models.py /app/scripts/model_usage/
COPY model_usage/model_utils.py /app/scripts/model_usage/

RUN chmod +x /app/scripts/model_usage/*.py

EXPOSE 8080

ENTRYPOINT ["java", "-jar", "FogNode-0.0.1-SNAPSHOT.jar"]