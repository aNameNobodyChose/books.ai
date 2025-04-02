import argparse
import json
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.cluster import KMeans

def read_story(filename):
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read()
    return text

def extract_dialogues(text):
    dialogue = re.findall(r'"(.*?)"', text)
    return dialogue

def map_patterns_responses(dialogues):
    return {
        dialogues[i]: dialogues[i + 1]
        for i in range(0, len(dialogues) - 1, 2)
    }

# Cluster sentences to infer intent
def cluster_intents(sentences, num_clusters=None, max_k=10, plot_elbow=True):
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(sentences)
    if num_clusters is None:
        max_k = min(max_k, len(sentences))
        inertias = []
        k_range = range(1, max_k + 1)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        if plot_elbow:
            plt.plot(k_range, inertias, marker='o')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Inertia')
            plt.title('Elbow Method for Optimal k')
            plt.grid(True)
            plt.savefig("elbow_plot.png")
        
        drops = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
        best_k = drops.index(max(drops)) + 1
        print(f"[INFO] Using optimal k from elbow method: {best_k}")
        num_clusters = best_k
    
    # Run KMeans with chosen or provided k
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    clustered_data = defaultdict(list)
    for i, sentence in enumerate(sentences):
        clustered_data[labels[i]].append(sentence)

    return clustered_data

def classify_intents(clustered_data, responses):
    classified_data = defaultdict(lambda: {"patterns": [], "responses": []})

    for cluster_id, queries in clustered_data.items():
        tag = f"intent_{cluster_id}"
        for query in queries:
            classified_data[tag]["patterns"].append(query)
            classified_data[tag]["responses"].append(responses[query])
    return classified_data

def save_to_json(classified_data, output_filename):
    training_data = {"intents": [
        {"tag": tag, "patterns": data["patterns"], "responses": data["responses"]}
        for tag, data in classified_data.items()
    ]}

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2)

    print(f"Training data saved to {output_filename}")

def main():
    parser = argparse.ArgumentParser(description="Process a story and write training data.")
    parser.add_argument("--input", required=True, help="Path to the input file")
    parser.add_argument("--output", required=True, help="Path to the output file")
    args = parser.parse_args()

    text = read_story(args.input)
    extracted_dialogues = extract_dialogues(text)

    pattern_to_responses = map_patterns_responses(extracted_dialogues)
    clustered_data = cluster_intents(list(pattern_to_responses.keys()), num_clusters=None, plot_elbow=True)

    classified_data = classify_intents(clustered_data, pattern_to_responses)

    save_to_json(classified_data, args.output)

# Run script
if __name__ == "__main__":
    main()