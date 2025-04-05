import re
import argparse

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

def main():
    parser = argparse.ArgumentParser(description="Pass story")
    parser.add_argument("--input", required=True, help="Path to the input file")
    parser.add_argument("--output", required=True, help="Path to the output file")
    args = parser.parse_args()
    dialogue_data = extract_dialogues_with_context_from_file(args.input)
    print(dialogue_data)

# Run script
if __name__ == "__main__":
    main()