#!/bin/bash

# Set the image name and script to run
IMAGE_NAME="pytorch-chatbot:latest"
SCRIPT_NAME="chat.py"

# Initialize variables
INTENTS_FILE=""
MODEL_FILE=""
BOT_NAME=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --intents)
      INTENTS_FILE="$2"
      shift
      shift
      ;;
    --model)
      MODEL_FILE="$2"
      shift
      shift
      ;;
    --bot-name)
      BOT_NAME="$2"
      shift
      shift
      ;;
    *)
      echo "❌ Unknown option: $1"
      echo "Usage: $0 --intents <intents_file> --model <model_file> --bot-name <bot_name>"
      exit 1
      ;;
  esac
done

# Check if required args are present
if [[ -z "$INTENTS_FILE" || -z "$MODEL_FILE" || -z "$BOT_NAME" ]]; then
  echo "❌ Error: --intents, --model, and --bot-name arguments are required."
  echo "Usage: $0 --intents <intents_file> --model <model_file> --bot-name <bot_name>"
  exit 1
fi

# Run the Docker container
docker run --rm -it \
  -v "$(pwd)":/workspace \
  -w /workspace \
  $IMAGE_NAME \
  python $SCRIPT_NAME --intents "$INTENTS_FILE" --model "$MODEL_FILE" --bot-name "$BOT_NAME"
