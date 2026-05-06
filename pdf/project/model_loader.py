from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import timm
from config import DEVICE, MODEL_FILES, CONTENT_CLASSES, EVIDENCE_CLASSES, QUALITY_CLASSES, TAMPER_CLASSES, INFERENCE_TTA, OOD_MAX_SOFTMAX, TAMPERING_THRESHOLD

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p), (1, 1)).pow(1.0 / self.p)

class CNN1(nn.Module):
    def __init__(self, nc=len(CONTENT_CLASSES)):
        super().__init__()
        self.backbone = timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=0, global_pool="")
        feat_dim = self.backbone.num_features
        self.pool = GeM()
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(feat_dim, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.3), nn.Linear(512, nc))
    def forward(self, x):
        return self.head(self.pool(self.backbone(x)))
    def embed(self, x):
        return self.pool(self.backbone(x)).flatten(1)

class CNN2(nn.Module):
    def __init__(self, embed_dim=1280, hidden=512, nc=len(EVIDENCE_CLASSES)):
        super().__init__()
        bb = models.resnext50_32x4d(weights=None)
        feat = bb.fc.in_features
        self.backbone = nn.Sequential(*list(bb.children())[:-1])
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.35), nn.Linear(feat + embed_dim, hidden), nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(0.25), nn.Linear(hidden, nc))
    def forward(self, x, e1):
        return self.head(torch.cat([self.backbone(x).flatten(1), e1], dim=1))

class CNN3(nn.Module):
    def __init__(self, hidden=256, nc=len(QUALITY_CLASSES)):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        feat = self.backbone.num_features
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat, hidden), nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(0.2), nn.Linear(hidden, nc))
    def forward(self, x):
        return self.head(self.backbone(x))

class CNN4(nn.Module):
    def __init__(self, hidden=512, nc=len(TAMPER_CLASSES)):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b1", pretrained=False, num_classes=0)
        feat = self.backbone.num_features
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(feat, hidden), nn.ReLU(), nn.BatchNorm1d(hidden), nn.Dropout(0.3), nn.Linear(hidden, nc))
    def forward(self, x):
        return self.head(self.backbone(x))

class ModelManager:
    def __init__(self):
        self.device = self._device()
        self.tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.models = {}
        self.errors = {}
        self.load_all()
    def _device(self):
        if DEVICE == "cpu":
            return torch.device("cpu")
        if DEVICE == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def _load_state(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        return ckpt.get("model_state", ckpt), float(ckpt.get("temperature", 1.0))
    def _finish(self, name, model, path):
        state, temp = self._load_state(path)
        model.load_state_dict(state)
        model.to(self.device).eval()
        self.models[name] = {"model": model, "temperature": temp, "path": str(path)}
    def _hidden(self, state):
        if "head.1.weight" in state:
            return int(state["head.1.weight"].shape[0])
        if "head.2.weight" in state:
            return int(state["head.2.weight"].shape[0])
        return 512
    def load_all(self):
        self.models = {}
        self.errors = {}
        for name, path in MODEL_FILES.items():
            if not Path(path).exists():
                self.errors[name] = "missing"
        try:
            if Path(MODEL_FILES["content"]).exists():
                self._finish("content", CNN1(), MODEL_FILES["content"])
        except Exception as e:
            self.errors["content"] = str(e)
        try:
            if Path(MODEL_FILES["quality"]).exists():
                state, temp = self._load_state(MODEL_FILES["quality"])
                model = CNN3(hidden=self._hidden(state))
                model.load_state_dict(state)
                model.to(self.device).eval()
                self.models["quality"] = {"model": model, "temperature": temp, "path": str(MODEL_FILES["quality"])}
        except Exception as e:
            self.errors["quality"] = str(e)
        try:
            if Path(MODEL_FILES["tamper"]).exists():
                state, temp = self._load_state(MODEL_FILES["tamper"])
                model = CNN4(hidden=self._hidden(state))
                model.load_state_dict(state)
                model.to(self.device).eval()
                self.models["tamper"] = {"model": model, "temperature": temp, "path": str(MODEL_FILES["tamper"])}
        except Exception as e:
            self.errors["tamper"] = str(e)
        try:
            if Path(MODEL_FILES["evidence"]).exists() and "content" in self.models:
                state, temp = self._load_state(MODEL_FILES["evidence"])
                dim = state["head.2.weight"].shape[1] - 2048
                model = CNN2(embed_dim=dim, hidden=self._hidden(state))
                model.load_state_dict(state)
                model.to(self.device).eval()
                self.models["evidence"] = {"model": model, "temperature": temp, "path": str(MODEL_FILES["evidence"])}
        except Exception as e:
            self.errors["evidence"] = str(e)
    def status(self):
        rows = []
        for name, path in MODEL_FILES.items():
            item = self.models.get(name)
            classes = {"content": CONTENT_CLASSES, "evidence": EVIDENCE_CLASSES, "quality": QUALITY_CLASSES, "tamper": TAMPER_CLASSES}.get(name, [])
            rows.append({"name": name, "loaded": item is not None, "path": str(path), "error": None if item else self.errors.get(name), "temperature": item.get("temperature") if item else None, "classes": classes, "tta": INFERENCE_TTA, "device": str(self.device)})
        return rows
    def _tensor(self, image: Image.Image):
        return self.tf(image.convert("RGB")).unsqueeze(0).to(self.device)
    def _forward(self, item, x, extra=None):
        if extra is None:
            return item["model"](x)
        return item["model"](x, extra)
    def _predict(self, name, x, classes, extra=None, extra_flip=None):
        item = self.models.get(name)
        if not item:
            return None
        with torch.inference_mode():
            logits = self._forward(item, x, extra)
            if INFERENCE_TTA:
                xf = torch.flip(x, dims=[3])
                logits = (logits + self._forward(item, xf, extra_flip if extra_flip is not None else extra)) / 2
            prob = F.softmax(logits / max(item["temperature"], 1e-6), dim=1).detach().cpu().numpy()[0]
        idx = int(prob.argmax())
        order = np.argsort(prob)[::-1]
        return {"label": classes[idx], "confidence": round(float(prob[idx]), 5), "runner_up": classes[int(order[1])] if len(order) > 1 else None, "scores": {classes[i]: round(float(prob[i]), 5) for i in range(len(classes))}}
    def classify_image(self, image: Image.Image):
        x = self._tensor(image)
        out = {}
        content = self._predict("content", x, CONTENT_CLASSES)
        if content:
            content["unclassifiable"] = bool(content["confidence"] < OOD_MAX_SOFTMAX)
            out["content"] = content
        quality = self._predict("quality", x, QUALITY_CLASSES)
        if quality:
            out["quality"] = quality
        tamper = self._predict("tamper", x, TAMPER_CLASSES)
        if tamper:
            tamper["risk_label"] = "possibly_tampered" if tamper["scores"].get("tampered", 0.0) >= TAMPERING_THRESHOLD else "likely_authentic"
            out["tamper"] = tamper
        if "evidence" in self.models and "content" in self.models:
            with torch.inference_mode():
                e = self.models["content"]["model"].embed(x)
                ef = self.models["content"]["model"].embed(torch.flip(x, dims=[3])) if INFERENCE_TTA else e
            evidence = self._predict("evidence", x, EVIDENCE_CLASSES, e, ef)
            if evidence:
                out["evidence"] = evidence
        return out
