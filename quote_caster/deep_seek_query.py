import requests

url = "http://localhost:11434/api/generate"

payload = {
    "model": "deepseek-r1",
    "prompt": "Hello, what can you do?",
    "stream": False
}

response = requests.post(url, json=payload)

# Parse and print the response
if response.ok:
    data = response.json()
    print("Model response:", data.get("response"))
else:
    print("Error:", response.status_code, response.text)