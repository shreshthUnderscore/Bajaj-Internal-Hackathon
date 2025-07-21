# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies required by faiss-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container at /app
COPY . .

# The command to run your app. Render will set the PORT environment variable.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
