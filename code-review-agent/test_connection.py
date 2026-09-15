import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "qwen2.5-coder:1.5b",
        "prompt": "Say hello in one short sentence.",
        "stream": False
    }
)

print(response.json()["response"])