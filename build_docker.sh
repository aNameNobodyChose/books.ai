#!/bin/bash

# Set the image name
IMAGE_NAME="pytorch-chatbot"

echo "Building Docker image: $IMAGE_NAME"

# Run the docker build command
docker build -t $IMAGE_NAME .

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Docker image '$IMAGE_NAME' built successfully!"
else
    echo "Docker build failed!" >&2
    exit 1
fi
