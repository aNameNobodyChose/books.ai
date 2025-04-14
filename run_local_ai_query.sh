#!/bin/bash

# Set the image name and script to run
IMAGE_NAME="pytorch-chatbot:latest"
SCRIPT_NAME="quote_caster.local_ai_query"

# Initialize variables
STORY_FILE=""
QUOTES_FILE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --story)
      STORY_FILE="$2"
      shift
      shift
      ;;
    --quotes)
      QUOTES_FILE="$2"
      shift
      shift
      ;;
    *)
      echo "❌ Unknown option: $1"
      echo "Usage: $0 --story <story_file> --quotes <quotes_file>"
      exit 1
      ;;
  esac
done

# Check if required args are present
if [[ -z "$STORY_FILE" || -z "$QUOTES_FILE" ]]; then
  echo "❌ Error: --story and --quotes arguments are required."
  echo "Usage: $0 --story <story_file> --quotes <quotes_file>"
  exit 1
fi

# Run the Docker container
docker run --rm -it \
  --network host \
  -v "$(pwd)":/workspace \
  -w /workspace \
  $IMAGE_NAME \
  python -m $SCRIPT_NAME --story "$STORY_FILE" --quotes "$QUOTES_FILE"
