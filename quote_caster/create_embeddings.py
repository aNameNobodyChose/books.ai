def embed_dataset(data, tokenizer, model) -> torch.Tensor:
    """
    Given a list of dicts with 'context' and 'quote', returns stacked quote embeddings.
    """
    embeddings = []
    for ex in data:
        emb = encode_quote(ex["context"], ex["quote"], tokenizer, model)
        embeddings.append(emb)

    return torch.stack(embeddings)  # shape: [num_quotes, hidden_dim]

def build_speaker_embeddings(data, quote_embeddings):
    """
    Builds an average embedding for each speaker.

    Args:
    - data: list of dicts with 'speaker'
    - quote_embeddings: torch.Tensor of shape [n_quotes, 768]

    Returns:
    - speaker_embeddings: dict of speaker_id -> avg torch.Tensor [768]
    """
    speaker_vectors = defaultdict(list)

    for i, ex in enumerate(data):
        speaker = ex["speaker"]
        speaker_vectors[speaker].append(quote_embeddings[i])

    speaker_embeddings = {
        speaker: torch.stack(vectors).mean(dim=0)
        for speaker, vectors in speaker_vectors.items()
    }

    return speaker_embeddings

def run_embedding_pipeline(anonymized_data):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    model.eval()

    embeddings = embed_dataset(anonymized_data, tokenizer, model)

    print("Quote embeddings shape:", embeddings.shape)
    return embeddings