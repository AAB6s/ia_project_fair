from functools import lru_cache
import torch
from config import LAYOUT_ENABLED, LAYOUT_MODEL_ID, LAYOUT_MIN_SCORE, DEVICE

class HuggingFaceLayoutDetector:
    def __init__(self, model_id):
        from transformers import pipeline
        if DEVICE == "cpu":
            device = -1
        elif DEVICE == "cuda":
            device = 0 if torch.cuda.is_available() else -1
        else:
            device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline("object-detection", model=model_id, device=device)

    def detect(self, image):
        rows = []
        for item in self.pipe(image):
            score = float(item.get("score", 0.0))
            if score < LAYOUT_MIN_SCORE:
                continue
            box = item.get("box") or {}
            rows.append({
                "label": str(item.get("label", "unknown")),
                "score": score,
                "bbox_px": [
                    float(box.get("xmin", 0.0)),
                    float(box.get("ymin", 0.0)),
                    float(box.get("xmax", 0.0)),
                    float(box.get("ymax", 0.0)),
                ],
            })
        return rows

@lru_cache(maxsize=1)
def get_layout_detector():
    if not LAYOUT_ENABLED or not LAYOUT_MODEL_ID:
        return None
    try:
        return HuggingFaceLayoutDetector(LAYOUT_MODEL_ID)
    except Exception:
        return None
