import streamlit as st
import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer

# Connect to LanceDB and open your table
db = lancedb.connect("my_lancedb")
table = db.open_table("notes")

# Load your embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")

st.title("LanceDB Notes Manager")

# Section: Insert new note
st.header("Insert New Note")

note_text = st.text_area("Enter note text")

if st.button("Insert Note"):
    if note_text.strip() == "":
        st.error("Note text cannot be empty!")
    else:
        # Generate embedding vector
        vector = model.encode([note_text]).tolist()[0]

        # Add the new note to LanceDB
        table.add([{"text": note_text, "vector": vector}])
        st.success("Note inserted successfully!")

# Section: View existing notes
st.header("View Existing Notes")

df = table.to_pandas()
st.dataframe(df[["text"]])
