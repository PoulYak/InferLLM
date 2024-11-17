# Use an NVIDIA base image with Python and CUDA support
FROM nvidia/cuda:11.4.3-base-ubuntu20.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    gcc \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip &&  \
    pip install --no-cache-dir -r requirements.txt && \
    pip install huggingface-hub

# Set up Hugging Face token for authentication
ARG HF_TOKEN
RUN huggingface-cli login --token $HF_TOKEN

# Copy application code
COPY main.py .

# Expose the FastAPI port
EXPOSE 8000

# Command to start the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]