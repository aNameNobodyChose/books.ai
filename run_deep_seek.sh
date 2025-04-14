#!/bin/bash

# Step 1: Pull the latest Ollama Docker image
echo "Pulling Ollama Docker image..."
docker pull ollama/ollama:0.6.5

# Step 2: Create and start the container
echo "Running Ollama container..."
docker run -d --name ollama \
  -p 11434:11434 \
  --pull=always \
  --volume ollama:/root/.ollama \
  ollama/ollama

# Step 3: Wait a few seconds for the container to fully start
echo "Waiting for Ollama to initialize..."
sleep 5

# Step 4: Pull the DeepSeek Chat model inside the container
echo "Pulling deepseek-chat model..."
docker exec ollama ollama pull deepseek-llm:7b

# Step 5: (Optional) Run a test API call from host
echo "Testing DeepSeek Chat API..."
curl -s http://localhost:11434/api/generate -d '{
  "model": "deepseek-llm:7b",
  "prompt": "Hello, what can you do?",
  "stream": false
}' | jq

echo -e "\n✅ DeepSeek Chat is ready and accessible at http://localhost:11434"
