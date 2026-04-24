import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = Path("data/processed/chunks.jsonl")
OUT_DIR = Path("data/processed")
INDEX_FILE = OUT_DIR / "faiss.index"
META_FILE = OUT_DIR / "meta.json"


model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

texts = []
meta = []

with CHUNKS_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        texts.append(obj["text"])
        meta.append(obj)

print(f"[INFO] loaded {len(texts)} chunks, encoding...")

emb = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,  # 用 cosine 相似度会更稳
)

emb = np.asarray(emb, dtype="float32")
dim = emb.shape[1]

# cosine 相似度：用 Inner Product + 归一化
index = faiss.IndexFlatIP(dim)
index.add(emb)

faiss.write_index(index, str(INDEX_FILE))
META_FILE.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

print(f"[OK] index -> {INDEX_FILE}")
print(f"[OK] meta  -> {META_FILE}")