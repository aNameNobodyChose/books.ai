#!/bin/bash

# Set the image name and the script path
IMAGE_NAME="pytorch-chatbot:latest"
SCRIPT_PATH="quote_caster/upload_to_hugging_face.py"

# Initialize variable
FILE_PATH=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --file_path)
      FILE_PATH="$2"
      shift
      shift
      ;;
    *)
      echo "❌ Unknown option: $1"
      echo "Usage: $0 --file_path <path_to_file>"
      exit 1
      ;;
  esac
done

# Check if required arg is present
if [[ -z "$FILE_PATH" ]]; then
  echo "❌ Error: --file_path argument is required."
  echo "Usage: $0 --file_path <path_to_file>"
  exit 1
fi

# Run the Docker container and execute the upload script
docker run --rm -it \
  --network host \
  -v "$(pwd)":/workspace \
  -w /workspace \
  -e HUGGINGFACE_TOKEN="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" \
  $IMAGE_NAME \
  python quote_caster/upload_to_hugging_face.py --file_path "$FILE_PATH"
