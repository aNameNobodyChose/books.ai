import re
from transformers import AutoTokenizer, AutoModel
import torch
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

@torch.no_grad()
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

def embed_dataset(data, tokenizer, model) -> torch.Tensor:
    """
    Given a list of dicts with 'context' and 'quote', returns stacked quote embeddings.
    """
    embeddings = []
    for ex in data:
        emb = encode_quote(ex["context"], ex["quote"], tokenizer, model)
        embeddings.append(emb)

    return torch.stack(embeddings)  # shape: [num_quotes, hidden_dim]

def run_embedding_pipeline(anonymized_data):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    model.eval()

    embeddings = embed_dataset(anonymized_data, tokenizer, model)

    print("Quote embeddings shape:", embeddings.shape)
    return embeddings


def main():
    HF_REPO_ID = "aNameNobodyChose/quote-speaker-attribution"

    # Customize this mapping based on your files in the repo
    SPLIT_FILES = {
        "umbrella": "umbrella.json",
    }

    datasets_dict = download_multiple_splits(HF_REPO_ID, SPLIT_FILES)
    for split_name, data in datasets_dict.items():
        anonymized, name_to_id, id_to_name = anonymize_speakers(data)
        quote_embeddings = run_embedding_pipeline(anonymized)
        print(quote_embeddings)


# Run script
if __name__ == "__main__":
    main()