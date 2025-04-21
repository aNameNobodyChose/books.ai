from transformers import AutoModel, AutoTokenizer
from quote_caster.quote_caster_model_training import encode_quote
import torch
from sklearn.cluster import KMeans
import json
import matplotlib.pyplot as plt
from kneed import KneeLocator

def load_unseen_story(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data)} quotes from: {json_path}")
    return data

def load_trained_encoder():
    tokenizer = AutoTokenizer.from_pretrained("aNameNobodyChose/quote-caster-encoder")
    model = AutoModel.from_pretrained("aNameNobodyChose/quote-caster-encoder")
    model.eval()  # Set to inference mode
    return tokenizer, model

def embed_dataset(data, tokenizer, model) -> torch.Tensor:
    """
    Given a list of dicts with 'context' and 'quote', returns stacked quote embeddings.
    """
    embeddings = []
    for ex in data:
        emb = encode_quote(ex["context"], ex["quote"], tokenizer, model)
        embeddings.append(emb)

    return torch.stack(embeddings)  # shape: [num_quotes, hidden_dim]

def run_embedding_pipeline(anonymized_data, tokenizer, model):
    model.eval()

    embeddings = embed_dataset(anonymized_data, tokenizer, model)

    print("Quote embeddings shape:", embeddings.shape)
    return embeddings

def auto_k_via_elbow(embeddings, max_k=10, output_file="elbow_auto.png"):
    """
    Automatically selects optimal k using the elbow method with KneeLocator.
    Returns the selected k and optionally saves the elbow plot.
    """
    X = embeddings.detach().numpy()
    inertias = []

    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    # Find the "elbow" point
    knee = KneeLocator(range(1, max_k + 1), inertias, curve="convex", direction="decreasing")
    optimal_k = knee.knee or 2  # Fallback to 2 if elbow not found

    # Plot and save
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, max_k + 1), inertias, marker='o')
    plt.axvline(x=optimal_k, color='r', linestyle='--', label=f"Elbow at k={optimal_k}")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method with Auto-k Detection")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)

    print(f"✅ Optimal k selected: {optimal_k} (saved plot to {output_file})")
    return optimal_k

def main():
    tokenizer, model = load_trained_encoder()
    new_story_data = load_unseen_story("./quotes.json")
    quote_embeddings = run_embedding_pipeline(new_story_data, tokenizer, model)
    print("✅ Embeddings shape:", quote_embeddings.shape)
    optimal_k = auto_k_via_elbow(quote_embeddings)
    labels = KMeans(n_clusters=optimal_k).fit_predict(quote_embeddings.detach().numpy())
    for quote, cluster_id in zip(new_story_data, labels):
        quote["predicted_speaker"] = f"SPEAKER_{cluster_id}"
    with open("./predicted_quotes.json", "w", encoding="utf-8") as f:
        json.dump(new_story_data, f, indent=2, ensure_ascii=False)
    print("✅ Speaker prediction complete. Results saved to predicted_quotes.json")

# Run script
if __name__ == "__main__":
    main()