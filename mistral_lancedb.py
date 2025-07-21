from sentence_transformers import SentenceTransformer
import lancedb
import requests

# Load embedding model once
model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to LanceDB and open table once
db = lancedb.connect("my_lancedb")
table = db.open_table("notes")

# Mistral API config
API_KEY = "lvvzcfsbq8P3LR4E8F7fTR9LL7GpJUG3"
API_URL = "https://api.mistral.ai/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

print("Ask anything (type 'exit' to quit):")

while True:
    # Get user input
    query = input("\nYou: ").strip()
    if query.lower() in ("exit", "quit"):
        print("Goodbye!")
        break

    # Embed and search LanceDB
    query_embedding = model.encode(query).tolist()
    results = table.search(query_embedding).limit(3).to_list()
    context = "\n".join([r["text"] for r in results])

    # Compose API request to Mistral
    payload = {
        "model": "mistral-tiny",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Use the context to answer the user's question."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        "temperature": 0.5
    }

    # Get response from Mistral
    response = requests.post(API_URL, headers=HEADERS, json=payload)

    if response.status_code == 200:
        answer = response.json()["choices"][0]["message"]["content"]
        print("\nAssistant:", answer)
    else:
        print("Error:", response.status_code, response.text)
