#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eenvoudige Streamlit-webapp voor DeltAI.
Start lokaal met:
    streamlit run deltai_app.py
Dan kun je in je browser situaties typen en de top-antwoorden zien.
"""
import os, json
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss

INDEX_DIR = "index"
META_PATH = os.path.join(INDEX_DIR, "meta.json")
FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")

@st.cache_resource
def load_index():
    if not os.path.exists(META_PATH) or not os.path.exists(FAISS_PATH):
        return None, None
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    index = faiss.read_index(FAISS_PATH)
    return meta, index

@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

st.title("🔍 DeltAI – interne kennisbank")
st.write("Typ hieronder een situatie of vraag van een huurder. DeltAI zoekt de meest relevante antwoorden uit de kennisbank (HTML + PDF + DOCX).")

meta, index = load_index()
if not meta or not index:
    st.warning("⚠️ Geen index gevonden. Run eerst `deltai_crawler.py` en `build_index.py`.")
else:
    model = load_model()
    query = st.text_input("Situatie of vraag")
    if query:
        with st.spinner("Zoeken..."):
            q_emb = model.encode([query], convert_to_numpy=True)
            D, I = index.search(q_emb, 5)
        st.subheader("Top 5 resultaten")
        for rank, idx in enumerate(I[0]):
            if idx >= len(meta["items"]): 
                continue
            it = meta["items"][idx]
            st.markdown(f"### {rank+1}. {it.get('q','(geen titel)')}")
            st.write(it.get("a",""))
            if it.get("sources"):
                st.markdown("**Bronnen:**")
                for s in it.get("sources", []):
                    st.markdown(f"- [{s}]({s})")
