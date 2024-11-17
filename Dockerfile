# Base image with Python and CUDA support
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Hugging Face CLI
RUN pip install huggingface-hub

# Set up Hugging Face token for authentication
ARG HF_TOKEN
RUN huggingface-cli login --token $HF_TOKEN

# Copy application code
COPY main.py .

# Expose the FastAPI port
EXPOSE 8000

# Command to start the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
