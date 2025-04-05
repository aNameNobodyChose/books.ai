# books.ai

# Build docker image
./build_docker.sh

# Extract Dialogues
./run_dialogue_extraction.sh --input ./sample_stories/umbrella.txt --output dialogues.txt

# Extract intents
./run_extract_intents.sh --input ./sample_stories/story.txt --output intents.json

# Train chatbot
./run_training.sh --input intents.json --output data.pth

# Run Chat Bot
./run_chatbot.sh --intents intents.json --model data.pth --bot-name barista
