import os
from datasets import load_dataset
from huggingface_hub import login

def download_multiple_splits(repo_id: str, split_files: dict):
    """
    Downloads a dataset from Hugging Face with custom split names mapped to file names.

    :param repo_id: Hugging Face dataset repo ID (e.g., 'aNameNobodyChose/quote-speaker-attribution')
    :param split_files: Dictionary mapping split names to filenames in the repo
    :return: A dataset dict with keys as split names
    """
    print(f"🔐 Authenticating to Hugging Face...")
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise ValueError("❌ HUGGINGFACE_TOKEN environment variable not set")
    login(token=token)

    print(f"📦 Downloading dataset from {repo_id} with splits: {split_files}")
    dataset = load_dataset(
        repo_id,
        data_files=split_files,
        token=token  # ensures private repo access
    )

    print(f"✅ Loaded splits: {list(dataset.keys())}")
    return dataset

if __name__ == "__main__":
    HF_REPO_ID = "aNameNobodyChose/quote-speaker-attribution"

    # Customize this mapping based on your files in the repo
    SPLIT_FILES = {
        "umbrella": "umbrella.json",
    }

    datasets_dict = download_multiple_splits(HF_REPO_ID, SPLIT_FILES)

    # Example: Print the first entry from each split
    for split_name, split_data in datasets_dict.items():
        print(f"\n🔹 First entry in split '{split_name}':")
        print(split_data[0])
