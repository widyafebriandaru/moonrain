import requests

API_KEY = "lvvzcfsbq8P3LR4E8F7fTR9LL7GpJUG3" 

url = "https://api.mistral.ai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Initial system prompt
messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]

print("Type your question (or 'exit' to quit):")

while True:
    user_question = input("\nYou: ")

    if user_question.strip().lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    # Append the user message
    messages.append({"role": "user", "content": user_question})

    payload = {
        "model": "mistral-tiny",
        "messages": messages,
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        reply = response.json()['choices'][0]['message']['content']
        print("\nAssistant:", reply)
        # Append the assistant's reply to the conversation history
        messages.append({"role": "assistant", "content": reply})
    else:
        print("Error:", response.status_code, response.text)
