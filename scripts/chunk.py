import json
import re
from pathlib import Path

TEXT_DIR = Path("data/processed/text")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "chunks.jsonl"

CHUNK_SIZE = 800
OVERLAP = 120
MIN_PARAGRAPH_LEN = 80


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_by_paragraph(text: str):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs


def merge_short_paragraphs(paragraphs):
    merged = []
    buffer = ""

    for p in paragraphs:
        if len(buffer) + len(p) < MIN_PARAGRAPH_LEN:
            buffer = (buffer + "\n" + p).strip()
        else:
            if buffer:
                merged.append(buffer)
            buffer = p

    if buffer:
        merged.append(buffer)

    return merged


def split_long_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start += chunk_size - overlap

    return chunks


def build_chunks(text: str):
    text = clean_text(text)
    paragraphs = split_by_paragraph(text)
    paragraphs = merge_short_paragraphs(paragraphs)

    final_chunks = []
    current = ""

    for p in paragraphs:
        if len(current) + len(p) + 2 <= CHUNK_SIZE:
            current = (current + "\n\n" + p).strip()
        else:
            if current:
                final_chunks.append(current)
            if len(p) <= CHUNK_SIZE:
                current = p
            else:
                split_chunks = split_long_text(p)
                final_chunks.extend(split_chunks[:-1])
                current = split_chunks[-1]

    if current:
        final_chunks.append(current)

    return final_chunks


with OUT_FILE.open("w", encoding="utf-8") as out:
    for txt_file in TEXT_DIR.glob("*.txt"):
        text = txt_file.read_text(encoding="utf-8", errors="ignore")
        chunks = build_chunks(text)

        for i, ch in enumerate(chunks):
            rec = {
                "text": ch,
                "source": txt_file.name.replace(".txt", ""),
                "chunk_id": i
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"[OK] chunks saved to {OUT_FILE}")