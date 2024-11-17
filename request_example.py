import requests

url = 'http://localhost:8000/generate'
payload = {
    "prompt":"Как у тебя дела?",
    "max_length":10,
    "temperature":1.0
}

response = requests.post(url, json=payload)
print(response.json())