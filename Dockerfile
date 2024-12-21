FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime

# Set working directory
WORKDIR /app

# Set non-interactive frontend for apt and preconfigure tzdata
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies (optional, if needed for your app)
RUN apt-get update && apt-get install -y \
    gcc wget git curl tzdata \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip (installed via conda) and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Hugging Face CLI
RUN pip install huggingface-hub

# Use Hugging Face token (if provided) for authentication
ARG HF_TOKEN
RUN huggingface-cli login --token $HF_TOKEN

# Copy application code
COPY main.py .

# Expose the FastAPI port
EXPOSE 8000

# Command to start the app (example: FastAPI app)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
