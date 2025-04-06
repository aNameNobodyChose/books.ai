import spacy
import re
import argparse
from collections import Counter
import coreferee

def extract_dialogues_with_context_from_file(file_path, context_window = 1):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    dialogue_data = []
    for i, line in enumerate(lines):
        quotes = re.findall(r'"([^"]+)"', line)
        for quote in quotes:
            context_lines = lines[max(0, i - context_window): i] + lines[i+1: i+1 + context_window]
            context = " ".join(context_lines)
            dialogue_data.append({
                "quote": quote,
                "context": context,
                "line_index": i
            })
    return dialogue_data

# NER(Named Entity recognition, you're going to write your own at some point)
def extract_character_names(story_text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(story_text)
    # Grab all PERSON entities
    names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

    # Normalize (remove duplicates like "Mia", "Mia.")
    normalized = [name.strip(".") for name in names]
    name_counts = Counter(normalized)

    # Return sorted character names by frequency
    return name_counts.most_common()

def resolve_coreferences(text):
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("coreferee")
    doc = nlp(text)

    coref_map = []

    for chain in doc._.coref_chains:
        mentions = list(chain)
        if not mentions:
            continue

        # Get main mention span
        main_tokens = mentions[0].token_indexes
        main_mention = doc[min(main_tokens):max(main_tokens)+1].text

        for mention in mentions[1:]:
            mention_tokens = mention.token_indexes
            mention_span = doc[min(mention_tokens):max(mention_tokens)+1].text
            if mention_span != main_mention:
                coref_map.append((mention_span, main_mention))
    return coref_map

def main():
    parser = argparse.ArgumentParser(description="Pass story")
    parser.add_argument("--input", required=True, help="Path to the input file")
    parser.add_argument("--output", required=True, help="Path to the output file")
    args = parser.parse_args()
    with open(args.input, "r", encoding="utf-8") as f:
        story_text = f.read()
    dialogue_data = extract_dialogues_with_context_from_file(args.input)
    character_names_by_frequency = extract_character_names(story_text)
    print(character_names_by_frequency)
    coref_links = resolve_coreferences(story_text)
    print(coref_links)
 
# Run script
if __name__ == "__main__":
    main()