# Use the official Python 3.10-slim image as a base
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the application will run on
EXPOSE 8000

# Create user
RUN groupadd -g 1601 aii_backend && \
    useradd -m -u 1601 -g aii_backend aii_backend && \
    chown -R aii_backend:aii_backend .


USER aii_backend

# Start the application using Gunicorn with Uvicorn worker
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "127.0.0.1:8002"]