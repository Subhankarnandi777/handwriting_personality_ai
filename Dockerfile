FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create input and output directories if they don't exist
RUN mkdir -p input/handwriting_images output/results output/reports models/pretrained models/ml_models

# Expose Streamlit and FastAPI ports
EXPOSE 8501
EXPOSE 8000
