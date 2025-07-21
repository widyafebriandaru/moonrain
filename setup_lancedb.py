import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer

# 1. Connect to or create a LanceDB database
db = lancedb.connect("my_lancedb")

# 2. Load a small embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Your simple sample documents
texts = [
    "I went hiking in the mountains yesterday.",
    "Today I learned about vector databases.",
    "LanceDB is really fast for local AI apps.",
    "My dog loves chasing squirrels in the park.",
]

# 4. Generate embeddings
embeddings = model.encode(texts).tolist()

# 5. Create a DataFrame with text + vector
data = pd.DataFrame({
    "text": texts,
    "vector": embeddings
})

# 6. Create or overwrite a table
table = db.create_table("notes", data=data, mode="overwrite")

print("✅ Inserted sample data into LanceDB.")
