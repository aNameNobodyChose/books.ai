#!/bin/bash

# Set the image name and script to run
IMAGE_NAME="pytorch-chatbot:latest"
SCRIPT_NAME="train.py"

# Initialize variables
INPUT_FILE=""
OUTPUT_FILE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --input)
      INPUT_FILE="$2"
      shift
      shift
      ;;
    --output)
      OUTPUT_FILE="$2"
      shift
      shift
      ;;
    *)
      echo "❌ Unknown option: $1"
      echo "Usage: $0 --input <input_file> --output <output_file>"
      exit 1
      ;;
  esac
done

# Check if required args are present
if [[ -z "$INPUT_FILE" || -z "$OUTPUT_FILE" ]]; then
  echo "❌ Error: --input and --output arguments are required."
  echo "Usage: $0 --input <input_file> --output <output_file>"
  exit 1
fi

# Run the Docker container
docker run --rm -it \
  -v "$(pwd)":/workspace \
  -w /workspace \
  $IMAGE_NAME \
  python $SCRIPT_NAME --input "$INPUT_FILE" --output "$OUTPUT_FILE"
