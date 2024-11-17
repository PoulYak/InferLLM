# Base image with CUDA 11.8 and Ubuntu 20.04
FROM nvidia/cuda:11.8.0-base-ubuntu20.04

# Set working directory
WORKDIR /app


# Install system dependencies and Python 3.11
RUN apt-get update && apt-get install -y \
    software-properties-common \
    gcc wget git curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-distutils \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

COPY requirements.txt .
RUN pip install --upgrade pip &&  \
    pip install --no-cache-dir -r requirements.txt && \
    pip install huggingface-hub \

ARG HF_TOKEN
RUN huggingface-cli login --token $HF_TOKEN

# Copy application code
COPY main.py .

# Expose the FastAPI port
EXPOSE 8000

# Command to start the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]