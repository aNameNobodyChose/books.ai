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

# Run quotes with context
./run_quotes_with_context.sh --input ./sample_stories/umbrella.txt --output quotes.json

# Run deepseek speaker attribution query locally
./run_deep_seek.sh

./run_local_ai_query.sh --story ./sample_stories/umbrella.txt --quotes quotes.json

# Run open ai speaker attribution query cloud
./run_cloud_ai_query.sh --story ./sample_stories/umbrella.txt --quotes quotes.json

# Upload quotes with speakers to hugging face
 ./run_upload_to_hugging_face.sh --file_path quotes_with_speakers_umbrella.json

# Download data from hugging face
./run_download_datat_from_hugging_face.sh

# Run quote caster training
./run_quote_caster_model_training.sh

# Run quote caster inference
./run_quote_caster_inference.sh

# Upload model to hugging face
./run_upload_to_hugging_face_model.sh --model_dir ./models/quote_encoder_all_stories

# Run Fast Api
./run_fast_api.sh

Example query: curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '[
        {
          "quote": "Forgot your umbrella too?",
          "context": "Mia turned to see a guy holding a dripping bag."
        },
        {
          "quote": "Yeah.",
          "context": "He smiled and shook his head."
        }
      ]'