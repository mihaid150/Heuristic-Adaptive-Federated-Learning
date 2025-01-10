# Use the arm64v8 Ubuntu 24.04 image
FROM arm64v8/ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

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

# Ensure the virtual environment is used for the rest of the commands
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app/scripts"

# Add the following line to ensure it's persisted in interactive shells
RUN echo "export PYTHONPATH=/app/scripts" >> /etc/profile.d/pythonpath.sh

# Suppress TensorFlow warnings by setting TF_CPP_MIN_LOG_LEVEL to 3
ENV TF_CPP_MIN_LOG_LEVEL=3


# Copy requirements.txt into the container
COPY requirements.txt /app/requirements.txt

# Activate the virtual environment and install dependencies
RUN /opt/venv/bin/pip install --upgrade pip setuptools wheel \
    && /opt/venv/bin/pip cache purge \
    && /opt/venv/bin/pip install -r /app/requirements.txt

# Set the working directory
WORKDIR /app

# Create necessary dirs
RUN mkdir -p /app/data/collected \
    && mkdir -p /app/data/clients \
    && mkdir -p /app/models/ \
    && mkdir -p /app/scripts/ \
    && mkdir -p /app/scripts/data_usage \
    && mkdir -p /app/scripts/model_usage \
    && mkdir -p /app/scripts/utils/ \
    && mkdir -p /app/evaluation

RUN mkdir -p /app/cache_json

# Copy the JAR file and Python scripts into the container
COPY EdgeNode-1.0-SNAPSHOT.jar /app/
COPY model_usage/__init__.py /app/scripts/model_usage/
COPY model_usage/client_env.py /app/scripts/model_usage/
COPY model_usage/model_evaluation.py /app/scripts/model_usage/
COPY data_usage/preprocessor.py /app/scripts/data_usage/
COPY data_usage/select_data.py /app/scripts/data_usage/
COPY data_usage/__init__.py /app/scripts/data_usage/
COPY utils/utils.py /app/scripts/utils/
COPY utils/__init__.py /app/scripts/utils/
COPY model_usage/create_init_lstm_model.py /app/scripts/model_usage/
COPY data_usage/select_multiple_data.py /app/scripts/data_usage/
COPY model_usage/genetics_training.py /app/scripts/model_usage/
COPY *.csv /app/data/collected/

# Make sure the scripts are executable in their respective directories
RUN chmod +x /app/scripts/data_usage/*.py \
    /app/scripts/model_usage/*.py \
    /app/scripts/utils/*.py

# Expose the port the app runs on
EXPOSE 8080

# Set the command to run the Java application
ENTRYPOINT ["java", "-jar", "EdgeNode-1.0-SNAPSHOT.jar"]
