import streamlit as st
import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer
import requests
import os
import easyocr
from PIL import Image
import numpy as np

# Initialize session state variables
if 'new_note' not in st.session_state:
    st.session_state.new_note = ""
if 'query' not in st.session_state:
    st.session_state.query = ""

# Load embedding model once
@st.cache_resource(show_spinner="Loading embedding model...")
def load_model_safely():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu", trust_remote_code=True)

# Load OCR model once
@st.cache_resource(show_spinner="Loading OCR model...")
def load_easyocr():
    return easyocr.Reader(['en', 'id'])  # support English + Indonesia

# Load models
model = load_model_safely()
ocr_reader = load_easyocr()

# Connect to LanceDB
db = lancedb.connect("my_lancedb")

# Create table if not exists
if "notes" not in db.table_names():
    db.create_table("notes", data=pd.DataFrame(columns=["text", "vector"]), mode="create")

table = db.open_table("notes")

# ✅ Sidebar: Add a new note
st.sidebar.header("📝 Add a New Note")
new_note = st.sidebar.text_area("Write your note here:", value=st.session_state.new_note, key="new_note_input")

if st.sidebar.button("Insert Note"):
    if new_note.strip():
        vector = model.encode(new_note).tolist()
        table.add(pd.DataFrame({
                "text": [new_note],
                "vector": [vector]
        })) 
        st.session_state.new_note = ""  # Clear the note input
        st.sidebar.success("Note inserted!")
        st.rerun()
    else:
        st.sidebar.warning("Please write something before inserting.")

# ✅ Sidebar: View & Delete all notes
st.sidebar.markdown("---")
st.sidebar.subheader("📄 All Notes")

try:
    df = table.to_pandas()
    for idx, row in df.iterrows():
        st.sidebar.markdown(f"- {row['text']}")
        if st.sidebar.button("🗑️ Delete", key=f"delete_{idx}"):
            table.delete(f"text = '{row['text'].replace("'", "''")}'")
            st.sidebar.success("Note deleted.")
            st.rerun()
except Exception as e:
    st.sidebar.error(f"Error loading notes: {e}")

# ✅ Main Panel: OCR Upload
st.header("🖼️ OCR from Image")
uploaded_file = st.file_uploader("Upload an image for OCR", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert PIL image to numpy
    img_array = np.array(image)

    # OCR processing
    with st.spinner("🔎 Extracting text from image..."):
        ocr_result = ocr_reader.readtext(img_array, detail=0)

    if ocr_result:
        extracted_text = "\n".join(ocr_result)
        st.subheader("📖 OCR Result")
        st.write(extracted_text)

        # Option to insert OCR text to notes
        if st.button("➕ Save OCR Result to Notes"):
            vector = model.encode(extracted_text).tolist()
            table.add(pd.DataFrame({
                    "text": [extracted_text],
                    "vector": [vector]
            }))
            st.success("OCR text saved to Notes!")
    else:
        st.warning("No text detected in the image.")

# ✅ Main Panel: Ask Assistant
st.header("🔍 Ask Assistant")

with st.form(key='query_form'):
    query = st.text_input("What do you want to know?", value=st.session_state.query, key="query_input")
    submitted = st.form_submit_button("Ask")
    
    if submitted and query:
        st.session_state.query = ""  # Clear the query input
        query_vector = model.encode(query).tolist()
        
        # Perform search and get results with scores
        results = table.search(query_vector).limit(5).to_list()
        
        # Convert to DataFrame and extract scores
        results_df = pd.DataFrame([{'text': item['text'], 'score': item['_distance']} for item in results])
        
        # Set similarity threshold
        SIMILARITY_THRESHOLD = 1.3  # L2 distance threshold
        
        # Filter results
        filtered_results = results_df[results_df['score'] <= SIMILARITY_THRESHOLD]
        
        st.subheader("📚 Retrieved Notes")
        
        if filtered_results.empty:
            st.write("No relevant notes found in database.")
            context = ""
        else:
            for note in filtered_results["text"]:
                st.markdown(f"- {note}")
            context = "\n".join(filtered_results["text"].tolist())

        # Call Mistral API
        API_KEY = os.getenv("MISTRAL_API_KEY", "lvvzcfsbq8P3LR4E8F7fTR9LL7GpJUG3")
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

        prompt = f"""Use the following notes to answer the question. If the answer is not in the notes, say "There is no notes related to that question" then answer the question from dataset you have as general AI. Always answer with Bahasa Indonesia.

Notes:
{context}

Question: {query}
Answer:"""

        payload = {
            "model": "mistral-tiny",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        }

        try:
            response = requests.post("https://api.mistral.ai/v1/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            answer = response.json()["choices"][0]["message"]["content"]
            st.subheader("💬 Assistant's Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"Error getting assistant response: {e}")
