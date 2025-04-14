import argparse
import json
import requests

def load_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_prompt(story, quote_context_data):
    return f"""You are given a short fictional story and a list of quotes with their surrounding context.

Your task is to determine **who is speaking each quote** based on the context and the story. For each item, add a "speaker" field that contains the speaker's name.

- Use **actual character names** found in the story.
- Infer the speaker of the dialogue even when not clearly stated.
- If the speaker is not inferable, label the speaker as "ambiguous".
- Do **not** use pronouns like "I", "he", "she" as speaker names.
- Do **not** invent character names that are not in the story.
- Keep all original fields (quote, context, line_index) and only add the new "speaker" field.

Here is the story:

\"\"\"{story}\"\"\"

Here is the list of quotes with context:

{json.dumps(quote_context_data, indent=2)}

Please return the list again with a "speaker" field added to each object, keeping all other fields as-is and preserving the original order.
"""

def query_deepseek(prompt, model="deepseek-llm:7b"):
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": model,
        "prompt": prompt,
        "stream": False
    })

    if response.ok:
        return response.json().get("response")
    else:
        raise RuntimeError(f"Request failed: {response.status_code} {response.text}")

def main():
    parser = argparse.ArgumentParser(description="Assign speakers to dialogue using DeepSeek via Ollama.")
    parser.add_argument("--story", required=True, help="Path to the story.txt file.")
    parser.add_argument("--quotes", required=True, help="Path to the quote_context_data.json file.")
    parser.add_argument("--output", default="quotes_with_speakers.json", help="Path to save the output JSON file.")
    args = parser.parse_args()

    # Load inputs
    story = load_file(args.story)
    quote_context_data = load_json(args.quotes)

    # Build prompt and query model
    prompt = build_prompt(story, quote_context_data)
    response_text = query_deepseek(prompt)

    try:
        # Try to parse response as JSON
        parsed = json.loads(response_text)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2)
        print(f"\n✅ Saved output to: {args.output}")
    except json.JSONDecodeError:
        print("\n⚠️ Failed to parse response as JSON. Raw output:")
        print(response_text)

if __name__ == "__main__":
    main()