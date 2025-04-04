import argparse
import json
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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
def cluster_intents(sentences, num_clusters=None, max_k=10, plot_silhouette=True):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X = vectorizer.fit_transform(sentences)

    if num_clusters is None:
        max_k = min(max_k, len(sentences) - 1)
        best_score = -1
        best_k = 2
        scores = []

        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            scores.append(score)
            if score > best_score:
                best_score = score
                best_k = k

        if plot_silhouette:
            plt.plot(range(2, max_k + 1), scores, marker='o')
            plt.xlabel("Number of Clusters (k)")
            plt.ylabel("Silhouette Score")
            plt.title("Silhouette Method for Optimal k")
            plt.grid(True)
            plt.savefig("silhouette_plot.png")

        print(f"[INFO] Using optimal k from silhouette method: {best_k}")
        num_clusters = best_k

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    clustered_data = defaultdict(list)
    for i, sentence in enumerate(sentences):
        clustered_data[labels[i]].append(sentence)

    return clustered_data

def classify_intents(clustered_data, responses):
    classified_data = defaultdict(lambda: {"patterns": [], "responses": []})

    # Flatten all sentences and keep cluster mapping
    all_sentences = []
    cluster_lookup = [] # which cluster each sentence lies in
    for cluster_id, queries in clustered_data.items():
        all_sentences.extend(queries)
        cluster_lookup.extend([cluster_id] * len(queries))
    
    # Vectorize all sentences
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1)
    X = vectorizer.fit_transform(all_sentences)
    feature_names = vectorizer.get_feature_names_out() # This gives you a list of all words

    cluster_word_scores = defaultdict(Counter)
    for i, cluster_id in enumerate(cluster_lookup):
        tfidf_vector = X[i].toarray()[0]
        for idx, score in enumerate(tfidf_vector):
            cluster_word_scores[cluster_id][feature_names[idx]] += score
    
    # Classify data with dynamic tags
    for cluster_id, queries in clustered_data.items():
        top_words = [word for word, _ in cluster_word_scores[cluster_id].most_common(3)]
        tag = "_".join(top_words[:2]) if top_words else f"intent_{cluster_id}"
        tag = tag.replace("_", " ").title().replace(" ", "_")  # Clean it up nicely

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
    clustered_data = cluster_intents(list(pattern_to_responses.keys()), num_clusters=None, plot_silhouette=True)

    classified_data = classify_intents(clustered_data, pattern_to_responses)

    save_to_json(classified_data, args.output)

# Run script
if __name__ == "__main__":
    main()