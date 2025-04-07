import spacy
import re
import argparse
from collections import Counter
from collections import defaultdict
import coreferee
from transformers import BertTokenizer
from dialogue_extraction.dataset import DialogueAttributionDataset
from sklearn.model_selection import train_test_split

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

def build_pronoun_resolution_map(coref_links):
    """
    Builds a map: pronoun -> set of resolved character names
    """
    pronoun_map = defaultdict(set)
    for mention, resolved in coref_links:
        mention = mention.lower().strip()
        resolved = resolved.strip()
        if mention in ['he', 'she', 'his', 'her']:
            pronoun_map[mention].add(resolved)
    return pronoun_map

# TODO: got to make this robust enough so that it doesn't mislabel the speaker.
def label_speakers(dialogues, character_names, coref_links):
    name_set = set([name for name, _ in character_names])
    pronoun_map = build_pronoun_resolution_map(coref_links)

    speech_verbs = ['said', 'asked', 'replied', 'told', 'whispered', 'shouted', 'murmured', 'added', 'sighed']

    for entry in dialogues:
        context = entry['context']
        context_lower = context.lower()
        speaker = "UNKNOWN"

        # Heuristic 1: Name + speech verb
        for name in name_set:
            for verb in speech_verbs:
                if f'{name.lower()} {verb}' in context_lower or f'{verb} {name.lower()}' in context_lower:
                    speaker = name
                    break
            if speaker != "UNKNOWN":
                break

        # Heuristic 2: Pronoun resolution (only if unambiguous)
        if speaker == "UNKNOWN":
            for pronoun in ['he', 'she', 'his', 'her']:
                if pronoun in context_lower:
                    resolved_names = pronoun_map.get(pronoun, set())
                    if len(resolved_names) == 1:
                        resolved_name = list(resolved_names)[0]
                        if resolved_name in name_set:
                            speaker = resolved_name
                            break

        entry['speaker'] = speaker

    return dialogues

def main():
    parser = argparse.ArgumentParser(description="Pass story")
    parser.add_argument("--input", required=True, help="Path to the input file")
    parser.add_argument("--output", required=True, help="Path to the output file")
    args = parser.parse_args()
    with open(args.input, "r", encoding="utf-8") as f:
        story_text = f.read()
    dialogue_data = extract_dialogues_with_context_from_file(args.input)
    character_names_by_frequency = extract_character_names(story_text)
    coref_links = resolve_coreferences(story_text)
    labeled_dialogues = label_speakers(dialogue_data, character_names_by_frequency, coref_links)

    # Clean up data. 
    clean_data = [d for d in labeled_dialogues if d['speaker'] != "UNKNOWN"]
    label2id = {name: idx for idx, name in enumerate(sorted(set(d['speaker'] for d in clean_data)))}
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Split data into training and testing data.
    train_data, test_data = train_test_split(clean_data, test_size=0.3, stratify=[d['speaker'] for d in clean_data])
    train_dataset = DialogueAttributionDataset(train_data, tokenizer, label2id)
    test_dataset = DialogueAttributionDataset(test_data, tokenizer, label2id)
    print(train_dataset.samples)

 
# Run script
if __name__ == "__main__":
    main()