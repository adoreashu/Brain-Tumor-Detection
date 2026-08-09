# Use an official Python runtime as a parent image (slim version to save space)
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required by OpenCV and TensorFlow
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code and models
# We copy app, models, and utils directories to maintain the expected structure
COPY app/ ./app/
COPY models/ ./models/
COPY utils/ ./utils/

# Expose port 8000 for the FastAPI server
EXPOSE 8000

# Set Python path so it can find the app modules
ENV PYTHONPATH=/app

# Command to run the application using Uvicorn
CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
