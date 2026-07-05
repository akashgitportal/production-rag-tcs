"""
Step 17 — Ingestion v2: adds per-row records for PDF financial tables.
Fixes the recall problem where a specific line (e.g. dividends) was diluted
inside a whole-table chunk. Overwrites chunks.jsonl so downstream steps
(embed, load, eval) run unchanged.
"""
import json
import uuid
from pathlib import Path
from collections import Counter

import pymupdf4llm
import openpyxl

DATA_DIR = "data"
OUTPUT_FILE = "chunks.jsonl"
MAX_CHARS = 1500


def doc_type_from_name(name: str) -> str:
    n = name.lower()
    if "annual" in n:
        return "annual_report"
    if "fact" in n:
        return "fact_sheet"
    return "data_sheet"


def clean_number(value: str) -> str:
    try:
        num = float(value)
        return str(int(num)) if num == int(num) else str(round(num, 2))
    except ValueError:
        return value


def make_chunk(content, chunk_type, source, doc_type, **extra):
    meta = {"source": source, "doc_type": doc_type, "chunk_type": chunk_type}
    meta.update(extra)
    return {"chunk_id": uuid.uuid4().hex[:12], "content": content, "metadata": meta}


# ---------- markdown table -> per-row records ------------------------------ #
def decompose_table(md_table: str, heading: str = "") -> list[str]:
    """Turn a markdown table into one focused record per data row:
    '[heading] RowLabel -> Col1: v1; Col2: v2'."""
    rows = [ln for ln in md_table.split("\n") if ln.strip().startswith("|")]
    parsed = []
    for ln in rows:
        cells = [c.strip().strip("*").strip() for c in ln.strip().strip("|").split("|")]
        parsed.append(cells)
    # drop markdown separator rows (|---|---|)
    parsed = [r for r in parsed if not all(set(c) <= set("-: ") for c in r)]
    if len(parsed) < 2:
        return []
    header = parsed[0]
    records = []
    for row in parsed[1:]:
        label = row[0]
        if not label:
            continue
        pairs = []
        for i in range(1, len(row)):
            val = row[i]
            if not val:
                continue
            col = header[i] if i < len(header) and header[i] else ""
            pairs.append(f"{col}: {val}" if col else val)
        if not pairs:
            continue
        prefix = f"[{heading}] " if heading else ""
        records.append(f"{prefix}{label} -> " + "; ".join(pairs))
    return records


# ---------- split a page into prose / table segments ----------------------- #
def segment_page(md: str):
    """Yield ('prose'|'table', content, nearest_heading)."""
    segments = []
    buf, mode, heading = [], "prose", ""
    for line in md.split("\n"):
        is_table = line.lstrip().startswith("|")
        if is_table and mode == "prose":
            if buf:
                segments.append(("prose", "\n".join(buf), heading))
            buf, mode = [line], "table"
        elif not is_table and mode == "table":
            segments.append(("table", "\n".join(buf), heading))
            buf, mode = [line], "prose"
            if line.strip().startswith("#"):
                heading = line.strip("# ").strip()
        else:
            buf.append(line)
            if not is_table and line.strip().startswith("#"):
                heading = line.strip("# ").strip()
    if buf:
        segments.append((mode, "\n".join(buf), heading))
    return segments


def pack_prose(text, max_chars=MAX_CHARS):
    blocks = [b.strip() for b in text.split("\n") if b.strip()]
    chunks, current = [], ""
    for block in blocks:
        if current and len(current) + len(block) + 1 > max_chars:
            chunks.append(current)
            current = block
        else:
            current = (current + "\n" + block).strip() if current else block
    if current:
        chunks.append(current)
    return chunks


def process_pdf(path: Path):
    doc_type = doc_type_from_name(path.name)
    pages = pymupdf4llm.to_markdown(str(path), use_ocr=False,
                                    page_chunks=True, show_progress=False)
    chunks = []
    for page in pages:
        text = page.get("text", "").strip()
        if not text:
            continue
        page_no = page.get("metadata", {}).get("page_number")
        for seg_type, content, heading in segment_page(text):
            if not content.strip():
                continue
            if seg_type == "table":
                # (a) keep the whole table for full-picture questions
                chunks.append(make_chunk(content, "table", path.name, doc_type,
                                         page=page_no))
                # (b) add one focused record per row for precise retrieval
                for rec in decompose_table(content, heading):
                    chunks.append(make_chunk(rec, "table_row", path.name,
                                             doc_type, page=page_no))
            else:
                for piece in pack_prose(content):
                    chunks.append(make_chunk(piece, "text", path.name,
                                             doc_type, page=page_no))
    return chunks


def process_excel(path: Path):
    doc_type = doc_type_from_name(path.name)
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    chunks = []
    for sheet in wb.worksheets:
        rows = []
        for row in sheet.iter_rows(values_only=True):
            cells = ["" if c is None else str(c).strip() for c in row]
            if any(cells):
                rows.append(cells)
        if len(rows) < 2:
            continue
        periods = rows[0]
        for row in rows[1:]:
            label = row[0]
            if not label:
                continue
            pairs = [f"{periods[i]}: {clean_number(row[i])}"
                     for i in range(1, len(row))
                     if i < len(periods) and row[i]]
            if not pairs:
                continue
            content = f"[{sheet.title}] {label} -> " + "; ".join(pairs)
            chunks.append(make_chunk(content, "table_row", path.name, doc_type,
                                     sheet=sheet.title, metric=label))
    wb.close()
    return chunks


def main():
    data_dir = Path(DATA_DIR)
    all_chunks = []
    for path in sorted(data_dir.iterdir()):
        ext = path.suffix.lower()
        if ext == ".pdf":
            print(f"Processing PDF   : {path.name}")
            all_chunks += process_pdf(path)
        elif ext == ".xlsx":
            print(f"Processing Excel : {path.name}")
            all_chunks += process_excel(path)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    by_type = Counter(c["metadata"]["chunk_type"] for c in all_chunks)
    print("\n" + "=" * 55)
    print(f"TOTAL chunks: {len(all_chunks)}  ->  {OUTPUT_FILE}")
    print("By type:", dict(by_type))
    print("=" * 55)


if __name__ == "__main__":
    main()
