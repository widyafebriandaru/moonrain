import streamlit as st
import lancedb
from sentence_transformers import SentenceTransformer

db = lancedb.connect("my_lancedb")
table = db.open_table("notes")
model = SentenceTransformer("all-MiniLM-L6-v2")

query = st.text_input("Ask something:")
if query:
    vec = model.encode(query).tolist()
    results = table.search(vec).limit(5).to_pandas()
    st.write(results)
