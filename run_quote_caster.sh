#!/bin/bash

# Set the image name and the script path
IMAGE_NAME="pytorch-chatbot:latest"
SCRIPT_NAME="quote_caster.quote_caster"

# Run the Docker container and execute the upload script
docker run --rm -it \
  --network host \
  -v "$(pwd)":/workspace \
  -w /workspace \
  -e HUGGINGFACE_TOKEN="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" \
  $IMAGE_NAME \
  python -m $SCRIPT_NAME
