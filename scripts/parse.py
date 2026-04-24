import os
from pathlib import Path

import fitz  # pymupdf
from docx import Document

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed/text")

OUT_DIR.mkdir(parents=True, exist_ok=True)


def pdf_to_text(path: Path) -> str:
    doc = fitz.open(str(path))
    parts = []
    for page in doc:
        parts.append(page.get_text())
    return "\n".join(parts)


def docx_to_text(path: Path) -> str:
    doc = Document(str(path))
    parts = []
    for p in doc.paragraphs:
        if p.text.strip():
            parts.append(p.text)
    # 如果你 docx 里表格很多，后面可以加“提取表格”的逻辑；先跑通主流程更重要
    return "\n".join(parts)


for file in RAW_DIR.iterdir():
    if not file.is_file():
        continue

    suffix = file.suffix.lower()
    try:
        if suffix == ".pdf":
            text = pdf_to_text(file)
        elif suffix == ".docx":
            text = docx_to_text(file)
        else:
            continue

        out_path = OUT_DIR / f"{file.name}.txt"
        out_path.write_text(text, encoding="utf-8")
        print(f"[OK] {file.name} -> {out_path}")
    except Exception as e:
        print(f"[FAIL] {file.name}: {e}")