import argparse
import json
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
def cluster_intents(sentences, num_clusters=3):
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(sentences) # This literally give you a bag of words
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
    clustered_data = cluster_intents(list(pattern_to_responses.keys()))

    classified_data = classify_intents(clustered_data, pattern_to_responses)

    save_to_json(classified_data, args.output)

# Run script
if __name__ == "__main__":
    main()