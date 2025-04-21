import re
import random
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import os
from typing import List, Dict, Tuple
from collections import defaultdict
from quote_caster.download_data_from_hugging_face import download_multiple_splits

def anonymize_speakers(
    data: List[Dict]
) -> Tuple[List[Dict], Dict[str, str], Dict[str, str]]:
    """
    Replaces speaker names with SPEAKER_X tokens in both dialogue and context,
    and returns the updated dataset along with mapping dictionaries.

    Parameters:
    - data: list of examples with keys 'quote', 'context', 'speaker'

    Returns:
    - anonymized_data: new dataset with SPEAKER_X in place of names
    - name_to_id: map from real name → SPEAKER_X
    - id_to_name: map from SPEAKER_X → real name
    """

    name_to_id = {}
    id_to_name = {}
    speaker_counter = 1

    all_names = set(entry["speaker"] for entry in data)

    for name in sorted(all_names):  # ensure deterministic order
        speaker_id = f"SPEAKER_{speaker_counter}"
        name_to_id[name] = speaker_id
        id_to_name[speaker_id] = name
        speaker_counter += 1

    anonymized_data = []

    for entry in data:
        quote = entry["quote"]
        context = entry["context"]
        speaker = entry["speaker"]
        speaker_id = name_to_id[speaker]

        # Replace speaker names in quote and context with SPEAKER_X
        for name, sid in name_to_id.items():
            # only replace whole words (avoid replacing substrings)
            pattern = r'\b{}\b'.format(re.escape(name))
            quote = re.sub(pattern, sid, quote)
            context = re.sub(pattern, sid, context)

        anonymized_data.append({
            "quote": quote,
            "context": context,
            "speaker": speaker_id,
            "line_index": entry.get("line_index", -1)  # keep line index if present
        })

    return anonymized_data, name_to_id, id_to_name

def encode_quote(context: str, dialogue: str, tokenizer, model) -> torch.Tensor:
    """
    Encode a single quote using [CLS] token from BERT.
    """
    text = f"{context} [SEP] {dialogue}"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
    return cls_embedding.squeeze(0)  # shape: [hidden_dim]

class TripletQuoteDataset(Dataset):
    def __init__(self, data):
        """
        Prepares a dataset of (anchor, positive, negative) samples for training.

        Args:
        - data: list of dicts with keys: 'quote', 'context', 'speaker'
        """
        self.by_speaker = defaultdict(list)
        for ex in data:
            self.by_speaker[ex["speaker"]].append(ex)

        # Filter out speakers with fewer than 2 examples
        self.by_speaker = {k: v for k, v in self.by_speaker.items() if len(v) >= 2}
        self.speakers = list(self.by_speaker.keys())

    def __len__(self):
        # Just return total number of quotes (rough proxy)
        return sum(len(v) for v in self.by_speaker.values())

    def __getitem__(self, idx):
        # 1. Choose anchor speaker
        anchor_speaker = random.choice(self.speakers)

        # 2. Choose anchor + positive from the same speaker
        positives = self.by_speaker[anchor_speaker]
        anchor, positive = random.sample(positives, 2)

        # 3. Choose negative from a different speaker
        negative_speaker = random.choice([s for s in self.speakers if s != anchor_speaker])
        negative = random.choice(self.by_speaker[negative_speaker])

        return anchor, positive, negative

def encode_example(example, tokenizer, model):
    return encode_quote(example["context"], example["quote"], tokenizer, model)

def collate_fn(batch):
    """
    Keeps (anchor, positive, negative) triplets as-is without converting them into tensors.
    """
    anchors, positives, negatives = zip(*batch)
    return list(anchors), list(positives), list(negatives)

def train_encoder(dataset, model, tokenizer, epochs=3, lr=2e-5, batch_size=4):
    """
    Trains a transformer-based quote encoder using triplet margin loss.

    Args:
    - dataset: TripletQuoteDataset
    - model: HuggingFace AutoModel
    - tokenizer: matching HuggingFace tokenizer
    - epochs: number of training epochs
    - lr: learning rate
    - batch_size: number of triplets per batch
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    for epoch in range(epochs):
        total_loss = 0.0

        for batch in dataloader:
            anchors, positives, negatives = batch  # Fixed unpacking
            optimizer.zero_grad()
            batch_loss = 0.0

            for anchor, positive, negative in zip(anchors, positives, negatives):
                a_vec = encode_example(anchor, tokenizer, model)
                p_vec = encode_example(positive, tokenizer, model)
                n_vec = encode_example(negative, tokenizer, model)

                loss = F.triplet_margin_loss(a_vec, p_vec, n_vec, margin=1.0)
                batch_loss += loss

            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()

        print(f"✅ Epoch {epoch + 1}/{epochs} | Total Loss: {total_loss:.4f}")

def save_model_and_tokenizer(model, tokenizer, save_dir="models/quote_encoder_umbrella"):
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"✅ Model and tokenizer saved to: {save_dir}")

def main():
    HF_REPO_ID = "aNameNobodyChose/quote-speaker-attribution"

    # Customize this mapping based on your files in the repo
    SPLIT_FILES = {
        "umbrella": "umbrella.json",
    }

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    datasets_dict = download_multiple_splits(HF_REPO_ID, SPLIT_FILES)
    for split_name, data in datasets_dict.items():
        anonymized, name_to_id, id_to_name = anonymize_speakers(data)
        triplet_dataset = TripletQuoteDataset(anonymized)
        train_encoder(triplet_dataset, model, tokenizer, epochs=10)

    save_model_and_tokenizer(model, tokenizer, save_dir="./models/quote_encoder_all_stories")

# Run script
if __name__ == "__main__":
    main()