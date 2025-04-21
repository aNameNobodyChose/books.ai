#!/bin/bash

# Set the image name and script path
IMAGE_NAME="pytorch-chatbot:latest"
SCRIPT_NAME="quote_caster.upload_to_hugging_face_model"

# Initialize variable
MODEL_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --model_dir)
      MODEL_DIR="$2"
      shift
      shift
      ;;
    *)
      echo "❌ Unknown option: $1"
      echo "Usage: $0 --model_dir <path_to_model_directory>"
      exit 1
      ;;
  esac
done

# Check if required argument is present
if [[ -z "$MODEL_DIR" ]]; then
  echo "❌ Error: --model_dir argument is required."
  echo "Usage: $0 --model_dir <path_to_model_directory>"
  exit 1
fi

# Run the Docker container and execute the upload script
docker run --rm -it \
  --network host \
  -v "$(pwd)":/workspace \
  -w /workspace \
  -e HUGGINGFACE_TOKEN="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" \
  $IMAGE_NAME \
  python -m $SCRIPT_NAME --model_dir "$MODEL_DIR"
