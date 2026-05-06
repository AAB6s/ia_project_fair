from pathlib import Path
from PIL import Image
import io
import json
import fitz
from docx import Document
from openpyxl import load_workbook
from utils import normalize_text
from config import OCR_ENABLED

try:
    import pytesseract
except Exception:
    pytesseract = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TEXT_EXTS = {".txt", ".md", ".csv", ".tsv"}

def ocr_image(image: Image.Image) -> str:
    if not OCR_ENABLED or pytesseract is None:
        return ""
    try:
        return normalize_text(pytesseract.image_to_string(image))
    except Exception:
        return ""

def read_text_file(path: Path) -> list[dict]:
    for enc in ["utf-8", "utf-16", "latin-1"]:
        try:
            text = path.read_text(encoding=enc, errors="ignore")
            return [{"text": normalize_text(text), "page": None, "image": None, "kind": "text"}]
        except Exception:
            pass
    return []

def read_pdf(path: Path) -> list[dict]:
    rows = []
    doc = fitz.open(path)
    for i, page in enumerate(doc):
        text = normalize_text(page.get_text("text") or "")
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        if not text:
            text = ocr_image(image)
        rows.append({"text": text, "page": i + 1, "image": image, "kind": "pdf_page"})
    doc.close()
    return rows

def read_docx(path: Path) -> list[dict]:
    doc = Document(path)
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            values = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if values:
                parts.append(" | ".join(values))
    return [{"text": normalize_text("\n".join(parts)), "page": None, "image": None, "kind": "docx"}]

def read_xlsx(path: Path) -> list[dict]:
    wb = load_workbook(path, read_only=True, data_only=True)
    rows = []
    for ws in wb.worksheets:
        parts = []
        for row in ws.iter_rows(values_only=True):
            values = [str(v) for v in row if v not in [None, ""]]
            if values:
                parts.append(" | ".join(values))
        if parts:
            rows.append({"text": normalize_text("\n".join(parts)), "page": ws.title, "image": None, "kind": "xlsx_sheet"})
    return rows

def recursive_json_text(obj, path="") -> list[dict]:
    rows = []
    if isinstance(obj, dict):
        page = obj.get("page", obj.get("pg"))
        element_id = str(obj.get("id", obj.get("element_id", path)))
        for key, value in obj.items():
            name = str(key).lower()
            if isinstance(value, str) and len(value.strip()) > 30 and any(x in name for x in ["text", "ocr", "caption", "content", "summary", "clause", "paragraph", "transcript"]):
                rows.append({"text": normalize_text(value), "page": page, "image": None, "kind": "json_text", "element_id": element_id, "field": key})
            elif isinstance(value, (dict, list)):
                rows.extend(recursive_json_text(value, f"{path}/{key}"))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            rows.extend(recursive_json_text(value, f"{path}/{i}"))
    return rows

def read_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    return recursive_json_text(data)

def read_image(path: Path) -> list[dict]:
    image = Image.open(path).convert("RGB")
    return [{"text": ocr_image(image), "page": None, "image": image, "kind": "image"}]

def read_document(path: str | Path) -> list[dict]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        rows = read_pdf(path)
    elif ext in {".docx", ".doc"}:
        rows = read_docx(path)
    elif ext in {".xlsx", ".xls"}:
        rows = read_xlsx(path)
    elif ext in TEXT_EXTS:
        rows = read_text_file(path)
    elif ext == ".json":
        rows = read_json(path)
    elif ext in IMAGE_EXTS:
        rows = read_image(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    for row in rows:
        row.setdefault("source_file", path.name)
        row.setdefault("element_id", None)
        row["text"] = normalize_text(row.get("text", ""))
    return rows
