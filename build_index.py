#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bouwt een semantische index van alle Q&A in data/qa.jsonl
Gebruikt een sentence-transformer voor situatie → beste antwoord zoeken.
Output:
  index/faiss.index
  index/meta.json
"""
import os, json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

DATA_QA = "data/qa.jsonl"
INDEX_DIR = "index"
os.makedirs(INDEX_DIR, exist_ok=True)

def load_qa(path):
    items=[]
    with open(path,encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try: obj=json.loads(line)
            except: continue
            q=obj.get("question") or ""
            a=obj.get("answer") or ""
            srcs=obj.get("sources") or []
            items.append({"q":q,"a":a,"sources":srcs})
    return items

def main():
    if not os.path.exists(DATA_QA):
        print("⚠️ Geen data/qa.jsonl gevonden. Run eerst deltai_crawler.py")
        return
    items=load_qa(DATA_QA)
    if not items:
        print("⚠️ Geen Q&A gevonden.")
        return

    print(f"Items: {len(items)}")
    model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts=[(it["q"]+" "+it["a"]) for it in items]
    print("Embedden...")
    embs=model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    dim=embs.shape[1]
    index=faiss.IndexFlatL2(dim)
    index.add(embs)
    faiss.write_index(index, os.path.join(INDEX_DIR,"faiss.index"))

    meta={"items":items}
    with open(os.path.join(INDEX_DIR,"meta.json"),"w",encoding="utf-8") as f:
        json.dump(meta,f,ensure_ascii=False)

    print(f"Klaar! Index met {len(items)} items opgeslagen in {INDEX_DIR}/")

if __name__=="__main__":
    main()
