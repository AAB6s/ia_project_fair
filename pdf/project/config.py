from pathlib import Path
import os

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(path):
        if not Path(path).exists():
            return
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")
MODELS_DIR = ROOT_DIR / "models"
STORAGE_DIR = ROOT_DIR / "storage"
UPLOAD_DIR = ROOT_DIR / "uploads"
INDEX_DIR = STORAGE_DIR / "vector_indexes"
LAW_DIR = STORAGE_DIR / "law_knowledge"

for path in [MODELS_DIR, STORAGE_DIR, UPLOAD_DIR, INDEX_DIR, LAW_DIR]:
    path.mkdir(parents=True, exist_ok=True)

DEVICE = os.getenv("APP_DEVICE", "auto")
OCR_ENABLED = os.getenv("OCR_ENABLED", "true").lower() == "true"
INFERENCE_TTA = os.getenv("INFERENCE_TTA", "true").lower() == "true"
OOD_MAX_SOFTMAX = float(os.getenv("OOD_MAX_SOFTMAX", "0.4"))
TAMPERING_THRESHOLD = float(os.getenv("TAMPERING_THRESHOLD", "0.5"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RAG_EMBEDDINGS = os.getenv("RAG_EMBEDDINGS", "false").lower() == "true"
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "10"))
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "220"))
TOP_K = int(os.getenv("TOP_K", "5"))
POOL_K = int(os.getenv("POOL_K", "20"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "3000"))

MODEL_FILES = {
    "content": MODELS_DIR / "cnn1_content.pt",
    "evidence": MODELS_DIR / "cnn2_evidence.pt",
    "quality": MODELS_DIR / "cnn3_quality.pt",
    "tamper": MODELS_DIR / "cnn4_tamper.pt",
}

CONTENT_CLASSES = ["injury_photo", "scene_damage", "screenshot", "typed_document", "correspondence", "handwritten", "medical_record", "data_chart", "signature_stamp", "irrelevant"]
EVIDENCE_CLASSES = ["primary_evidence", "secondary_evidence"]
QUALITY_CLASSES = ["clear", "acceptable", "degraded"]
TAMPER_CLASSES = ["authentic", "tampered"]
