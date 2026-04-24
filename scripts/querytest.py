import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INDEX_PATH = "data/processed/faiss.index"
META_PATH = "data/processed/meta.json"

TOP_K = 6
MIN_SCORE = 0.25  # cosine相似度阈值，越大越严格

model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(INDEX_PATH)

with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

def retrieve(q: str, top_k=TOP_K):
    q_emb = model.encode([q], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")
    scores, idxs = index.search(q_emb, top_k)

    hits = []
    for s, i in zip(scores[0], idxs[0]):
        if i < 0:
            continue
        hits.append((float(s), meta[i]))
    return hits

while True:
    q = input("\n请输入问题（回车退出）：").strip()
    if not q:
        break

    hits = retrieve(q)

    if not hits or hits[0][0] < MIN_SCORE:
        print("\n⚠️ 资料未覆盖：我在现有教务资料中没有找到足够相关的依据。")
        print("建议：联系学院教务办/教务处，或提供相关通知截图让我补充到知识库。")
        continue

    print("\n🔍 检索到的依据（按相关度排序）：\n")
    for rank, (score, h) in enumerate(hits, 1):
        print(f"[{rank}] score={score:.3f}  来源={h['source']}  chunk={h['chunk_id']}")
        print(h["text"][:300].replace("\n", " "))
        print("-" * 60)