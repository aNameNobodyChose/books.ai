#!/bin/bash

# Set the image name and FastAPI entrypoint module
IMAGE_NAME="pytorch-chatbot:latest"
APP_MODULE="quote_caster.app:app"  # Format: <filename>:<FastAPI object>

# Run the Docker container with uvicorn serving FastAPI
docker run --rm -it \
  -p 8000:8000 \
  -v "$(pwd)":/workspace \
  -w /workspace \
  -e HUGGINGFACE_TOKEN="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" \
  $IMAGE_NAME \
  uvicorn "$APP_MODULE" --host 0.0.0.0 --port 8000 --reload
