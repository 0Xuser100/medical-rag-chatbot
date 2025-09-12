## Parent image
FROM python:3.12-slim

## Essential environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

## Work directory inside the docker container
WORKDIR /app

## Installing system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

## Copy requirements file first (for better Docker layer caching)
COPY app/requirements.txt .

## Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

## Copy the rest of the application
COPY app/ .

## Expose only flask port
EXPOSE 5000

## Run the Flask app
CMD ["python", "application.py"]


