import lancedb
from sentence_transformers import SentenceTransformer

# Connect to DB and open the table
db = lancedb.connect("my_lancedb")
table = db.open_table("notes")

# Load embedding model again
model = SentenceTransformer("all-MiniLM-L6-v2")

# Query example
query = "Animal"
query_vector = model.encode(query).tolist()

# Search for top 3 similar entries
results = table.search(query_vector).limit(1).to_pandas()

print("🔍 Search Results:")
print(results["text"])
