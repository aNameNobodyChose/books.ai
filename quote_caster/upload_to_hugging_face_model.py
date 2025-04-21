import argparse
import os
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import login, create_repo

def push_model_to_hf(model_dir, repo_id):
    # Get Hugging Face token from environment
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise ValueError("❌ HUGGINGFACE_TOKEN environment variable not set.")

    # Login to Hugging Face
    login(token=token)

    # Create repo if it doesn't exist
    create_repo(repo_id, repo_type="model", exist_ok=True)

    # Load and push tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.push_to_hub(repo_id)

    # Load and push model
    model = AutoModel.from_pretrained(model_dir)
    model.push_to_hub(repo_id)

    print(f"✅ Model and tokenizer pushed to: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload model directory to Hugging Face model hub.")
    parser.add_argument("--model_dir", required=True, help="Local path to the model directory")
    args = parser.parse_args()

    HF_REPO_ID = "aNameNobodyChose/quote-caster-encoder"
    push_model_to_hf(args.model_dir, HF_REPO_ID)
