import lancedb
# ... (previous imports remain the same)
import pyarrow as pa  # Make sure to import pyarrow

# Connect to LanceDB
db = lancedb.connect("my_lancedb")

# ✅ Create table with proper schema if not exists
if "notes" not in db.table_names():
    # Define proper schema with vector type
    schema = pa.schema([
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), list_size=384))  # 384 is the vector size for all-MiniLM-L6-v2
    ])
    db.create_table("notes", schema=schema)

table = db.open_table("notes")