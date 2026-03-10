"""Document parser for piRAG ingestion pipeline.

Handles .txt, .json, .csv formats with optional .pdf support.
Returns structured (doc_id, text, metadata) tuples for downstream
embedding and indexing.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple


def parse_document(path: str) -> Tuple[str, str, Dict[str, Any]]:
    """Parse a document file and extract its text content.

    Parameters
    ----------
    path : file path to parse.

    Returns
    -------
    (doc_id, text, metadata) where doc_id is derived from the filename,
    text is the extracted content, and metadata includes format and size info.
    """
    p = Path(path)
    doc_id = p.stem
    ext = p.suffix.lower()
    metadata: Dict[str, Any] = {
        "source": str(p.name),
        "format": ext.lstrip("."),
        "size_bytes": os.path.getsize(path),
    }

    if ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            text = json.dumps(data, indent=2)
            metadata["keys"] = list(data.keys())
        elif isinstance(data, list):
            text = "\n".join(str(item) for item in data)
            metadata["item_count"] = len(data)
        else:
            text = str(data)
    elif ext == ".csv":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(", ".join(row))
        text = "\n".join(rows)
        metadata["row_count"] = len(rows)
    elif ext == ".pdf":
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(path)
            pages = [page.get_text() for page in doc]
            text = "\n\n".join(pages)
            metadata["page_count"] = len(pages)
            doc.close()
        except ImportError:
            text = f"[PDF parsing unavailable for {p.name} - install PyMuPDF for PDF support]"
            metadata["parse_error"] = "PyMuPDF not installed"
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

    return doc_id, text.strip(), metadata
