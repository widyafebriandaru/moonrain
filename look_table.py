import streamlit as st
import lancedb
import pandas as pd

db = lancedb.connect("my_lancedb")
table = db.open_table("notes")
df = table.to_pandas()

st.title("LanceDB Notes Viewer")
st.write(df)
