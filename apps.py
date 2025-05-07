import streamlit as st
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
import re

# --- Streamlit UI ---
st.set_page_config(page_title="Dementia Detection Inference", layout="centered")
st.title("🧠 Dementia Detection - Inference App")

# --- Upload .cha file ---
cha_file = st.file_uploader("Upload a .cha Transcript File", type=["cha"])
model_file = st.file_uploader("Upload a .pkl Model File", type=["pkl"])

if cha_file and model_file:
    # --- Parse .cha file ---
    content = cha_file.read().decode("utf-8", errors="ignore")
    utterances = re.findall(r'\*PAR:\s(.+)', content)
    
    if not utterances:
        st.error("No participant utterances (*PAR:) found in the .cha file.")
    else:
        # --- Generate Sentence Embeddings ---
        st.info("Generating sentence embeddings...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(utterances)
        avg_embedding = embeddings.mean(axis=0).reshape(1, -1)

        # --- Load classifier model ---
        clf = joblib.load(model_file)

        # --- Predict ---
        pred = clf.predict(avg_embedding)[0]
        prob = clf.predict_proba(avg_embedding)[0] if hasattr(clf, "predict_proba") else None

        # --- Show Results ---
        st.success(f"🧾 Prediction: {'Dementia' if pred == 1 else 'Non-Dementia'}")
        if prob is not None:
            st.write(f"🧪 Prediction Probabilities: {prob}")
