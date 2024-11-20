# Base image with CUDA 11.8 and Ubuntu 22.04
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Set working directory
WORKDIR /app

# Set non-interactive frontend for apt and preconfigure tzdata
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies (без установки Python, он предустановлен)
RUN apt-get update && apt-get install -y \
    gcc wget git curl tzdata \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip for the preinstalled Python version
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Hugging Face CLI
RUN pip install huggingface-hub
ARG HF_TOKEN
RUN huggingface-cli login --token $HF_TOKEN

# Copy application code
COPY main.py .

# Expose the FastAPI port
EXPOSE 8000

# Command to start the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
