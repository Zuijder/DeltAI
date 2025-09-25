#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zoek een antwoord op basis van een situatie.
Gebruikt de semantische index uit build_index.py (faiss + meta.json).
Voorbeeld:
    python deltai_ask.py "Huurder wil laminaat leggen en vraagt of ondervloer verplicht is"
"""
import sys, json, os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

INDEX_DIR = "index"
META_PATH = os.path.join(INDEX_DIR, "meta.json")
FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")

def load_index():
    if not os.path.exists(META_PATH) or not os.path.exists(FAISS_PATH):
        print("⚠️ Index ontbreekt. Run eerst build_index.py")
        sys.exit(1)
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    index = faiss.read_index(FAISS_PATH)
    return meta, index

def main():
    if len(sys.argv) < 2:
        print("Geef een vraag of situatie als argument.")
        print('Voorbeeld: python deltai_ask.py "Ik wil laminaat leggen, is een ondervloer verplicht?"')
        return
    query = " ".join(sys.argv[1:])
    meta, index = load_index()
    items = meta.get("items", [])
    if not items:
        print("Geen items gevonden in meta.json")
        return

    print(f"Zoeken naar: {query}")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    q_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, 5)  # top 5 resultaten

    for rank, idx in enumerate(I[0]):
        if idx >= len(items): continue
        it = items[idx]
        print(f"\n--- #{rank+1} ---")
        print("Vraag/Titel:", it.get("q",""))
        print("Antwoord:", it.get("a",""))
        print("Bronnen:")
        for s in it.get("sources", []):
            print(" ", s)

if __name__ == "__main__":
    main()
