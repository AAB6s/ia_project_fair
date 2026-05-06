from pathlib import Path
import hashlib
import re
import uuid
import json

SAFE_NAME = re.compile(r"[^a-zA-Z0-9._-]+")

def clean_filename(name: str) -> str:
    stem = SAFE_NAME.sub("_", Path(name).stem).strip("_") or "file"
    suffix = SAFE_NAME.sub("", Path(name).suffix.lower())
    return f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"

def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def normalize_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def write_json(path: str | Path, data) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
