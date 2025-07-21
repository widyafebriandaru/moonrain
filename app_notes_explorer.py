import streamlit as st
import lancedb
from sentence_transformers import SentenceTransformer

@st.cache_resource(show_spinner="Loading embedding model...")
def load_model_safely():
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu", trust_remote_code=True)
    return model

# Load LanceDB
db = lancedb.connect("my_lancedb")
table = db.open_table("notes")

# Sidebar: insert a new note
st.sidebar.header("📝 Add a New Note")
new_note = st.sidebar.text_area("Write your note here:")

if st.sidebar.button("Insert Note"):
    if new_note.strip() != "":
        model = load_model_safely()
        vec = model.encode(new_note).tolist()
        table.add([{"text": new_note, "vector": vec}])
        st.sidebar.success("Note inserted!")
    else:
        st.sidebar.warning("Please enter a note before inserting.")

# Sidebar: show all notes
st.sidebar.markdown("---")
st.sidebar.subheader("📄 All Notes")
try:
    all_notes = table.to_pandas()
    st.sidebar.dataframe(all_notes[["text"]])
except Exception as e:
    st.sidebar.error(f"Error loading notes: {e}")

# Main panel: search interface
st.header("🔍 Explore Notes")
query = st.text_input("Ask something:")
if query:
    model = load_model_safely()
    vec = model.encode(query).tolist()
    results = table.search(vec).limit(1).to_pandas()
    st.dataframe(results)
