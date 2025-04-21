import argparse
import os
from huggingface_hub import login, create_repo, upload_file

def push_file_to_hf(file_path, repo_id):
    # Get the Hugging Face token from the environment
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise ValueError("❌ HUGGINGFACE_TOKEN environment variable not set.")

    # Login to Hugging Face
    login(token=token)

    # Validate file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")

    # Create repo if it doesn't exist
    create_repo(repo_id, repo_type="dataset", exist_ok=True)

    # Upload file to the repo
    upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.basename(file_path),
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )

    print(f"✅ Successfully uploaded `{file_path}` to {repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload dataset file to Hugging Face.")
    parser.add_argument("--file_path", required=True, help="Path to the dataset file (e.g., dialogues.json)")
    args = parser.parse_args()

    HF_REPO_ID = "aNameNobodyChose/quote-speaker-attribution"
    push_file_to_hf(args.file_path, HF_REPO_ID)
