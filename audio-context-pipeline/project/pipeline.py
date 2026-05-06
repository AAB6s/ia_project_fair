import hashlib
import json
import math
import os
import time
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
MODEL_PATH = APP_DIR / "models" / "resnet34_final.pt"
OUTPUT_DIR = ROOT_DIR / "audio_resnet_final_vf"
METRICS_PATH = OUTPUT_DIR / "final_metrics.json"
SUMMARY_PATH = OUTPUT_DIR / "final_summary.json"
XAI_PATH = OUTPUT_DIR / "xai" / "xai_manifest.json"


def load_local_env():
    for path in (APP_DIR / ".env", ROOT_DIR / ".env"):
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value


load_local_env()
EMOTION_MODEL_ID = os.getenv("AUDIO_EMOTION_MODEL_ID", "superb/wav2vec2-large-superb-er")
DEEPFAKE_MODEL_ID = os.getenv("AUDIO_DEEPFAKE_MODEL_ID", "Vansh180/deepfake-audio-wav2vec2")
PYANNOTE_MODEL_ID = os.getenv("AUDIO_DIARIZATION_MODEL_ID", "pyannote/speaker-diarization-3.1")
HF_TOKEN = os.getenv("HF_TOKEN", os.getenv("HUGGINGFACE_TOKEN", "")).strip()
HF_AUDIO_CLASSIFIERS = {}
WHISPER_MODELS = {}

TARGET_CLASSES = [
    "gunshot",
    "glass_break",
    "alarm_signal",
    "human_voice",
    "baby_cry",
    "background",
]

CFG = {
    "sample_rate": 32000,
    "segment_duration": 3.0,
    "segment_hop": 1.5,
    "n_mels": 96,
    "image_size": 224,
    "temperature": 0.7,
    "tta_shifts": [-8, 0, 8],
    "speech_min_duration": 1.0,
    "low_energy_rms": 0.0025,
    "speaker_threshold": 0.18,
}


class ModelUnavailableError(RuntimeError):
    pass


class ResNet34Spectrogram(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.backbone = models.resnet34(weights=None, progress=False)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Dropout(0.38),
            nn.Linear(in_features, 384),
            nn.ReLU(),
            nn.BatchNorm1d(384),
            nn.Dropout(0.28),
            nn.Linear(384, len(classes)),
        )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.cam_layer = self.backbone.layer4[-1].conv2

    def forward(self, spec):
        x = spec.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std
        return self.head(self.backbone(x))


def read_json(path, fallback=None):
    try:
        if Path(path).exists():
            return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return fallback
    return fallback


def file_sha256(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def available_model_path():
    candidates = [
        APP_DIR / "models" / "resnet34_final.pt",
        ROOT_DIR / "audio_resnet_final_vf" / "models" / "resnet34_final.pt",
    ]
    for path in candidates:
        if path.exists() and path.stat().st_size > 1024 * 1024:
            return path
    return candidates[0]


def model_status(path=None):
    path = Path(path or available_model_path())
    metrics = read_json(METRICS_PATH, {})
    status = {
        "model_path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "ready": path.exists() and path.stat().st_size > 1024 * 1024,
        "classes": TARGET_CLASSES,
        "metrics": metrics,
    }
    if status["exists"] and not status["ready"]:
        status["message"] = "Checkpoint exists but is empty or incomplete. Replace it with the real resnet34_final.pt."
    elif not status["exists"]:
        status["message"] = "Checkpoint missing. Place resnet34_final.pt in project/models."
    else:
        status["message"] = "Ready"
    return status


@lru_cache(maxsize=1)
def optional_runtime_status():
    status = {}
    try:
        import whisper
        status["whisper"] = {"available": True, "model": "tiny/base/small"}
    except Exception as exc:
        status["whisper"] = {"available": False, "reason": str(exc)}
    try:
        import transformers
        status["huggingface_audio"] = {
            "available": True,
            "emotion_model": EMOTION_MODEL_ID,
            "deepfake_model": DEEPFAKE_MODEL_ID,
        }
    except Exception as exc:
        status["huggingface_audio"] = {"available": False, "reason": str(exc)}
    try:
        import pyannote.audio
        status["pyannote"] = {
            "available": bool(HF_TOKEN),
            "installed": True,
            "model": PYANNOTE_MODEL_ID,
            "token": "set" if HF_TOKEN else "missing",
        }
    except Exception as exc:
        status["pyannote"] = {"available": False, "installed": False, "reason": str(exc)}
    try:
        import speechbrain
        status["speechbrain"] = {"available": True, "mode": "optional reference-speaker support"}
    except Exception as exc:
        status["speechbrain"] = {"available": False, "reason": str(exc)}
    return status


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def clean_state_dict(state):
    output = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("_orig_mod.", "module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        output[new_key] = value
    return output


def load_audio(path, sample_rate):
    try:
        waveform, source_rate = sf.read(str(path), dtype="float32", always_2d=False)
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)
        if int(source_rate) != int(sample_rate):
            waveform = resample_audio(waveform.astype(np.float32), int(source_rate), sample_rate)
    except Exception:
        waveform, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    if waveform.size == 0:
        raise ValueError("Empty audio file.")
    return waveform.astype(np.float32)


def resample_audio(y, orig_sr, target_sr):
    if int(orig_sr) == int(target_sr):
        return y.astype(np.float32)
    try:
        from scipy.signal import resample_poly
        common = math.gcd(int(orig_sr), int(target_sr))
        return resample_poly(y, int(target_sr) // common, int(orig_sr) // common).astype(np.float32)
    except Exception:
        return librosa.resample(y.astype(np.float32), orig_sr=int(orig_sr), target_sr=int(target_sr)).astype(np.float32)


def estimate_snr(y, sr):
    frame = max(256, int(0.025 * sr))
    hop = max(128, int(0.010 * sr))
    if len(y) < frame:
        return 0.0
    shape = (1 + (len(y) - frame) // hop, frame)
    strides = (y.strides[0] * hop, y.strides[0])
    frames = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)
    energy = np.mean(frames**2, axis=1)
    if len(energy) < 4:
        return 0.0
    sorted_energy = np.sort(energy)
    noise = float(np.mean(sorted_energy[: max(1, len(sorted_energy) // 10)]))
    signal = float(np.mean(sorted_energy[max(1, len(sorted_energy) // 10) :]))
    if noise <= 1e-10:
        return 60.0
    return float(10 * np.log10((signal + 1e-10) / noise))


def segment_audio(y, sr):
    segment_len = int(CFG["segment_duration"] * sr)
    hop_len = int(CFG["segment_hop"] * sr)
    if len(y) <= segment_len:
        padded = np.pad(y, (0, max(0, segment_len - len(y)))).astype(np.float32)
        return [{"start": 0.0, "end": round(len(y) / sr, 3), "waveform": padded, "padded": len(y) < segment_len}]
    starts = list(range(0, len(y) - segment_len + 1, hop_len))
    final_start = len(y) - segment_len
    if starts[-1] != final_start:
        starts.append(final_start)
    segments = []
    for start in starts:
        end = min(start + segment_len, len(y))
        segments.append(
            {
                "start": round(start / sr, 3),
                "end": round(end / sr, 3),
                "waveform": y[start : start + segment_len].astype(np.float32),
                "padded": False,
            }
        )
    return segments


def audio_to_spec(y):
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = y / max(1.0, peak)
    mel = mel_spectrogram(y)
    ref = float(np.max(mel)) if float(np.max(mel)) > 1e-12 else 1.0
    db = power_to_db(mel, ref=ref, top_db=80)
    spec = np.clip((db + 80) / 80, 0, 1).astype(np.float32)
    tensor = torch.tensor(spec).unsqueeze(0)
    tensor = F.interpolate(
        tensor.unsqueeze(0),
        size=(CFG["image_size"], CFG["image_size"]),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return tensor


@lru_cache(maxsize=1)
def mel_basis():
    sr = CFG["sample_rate"]
    n_fft = 2048
    n_mels = CFG["n_mels"]
    f_min = 0.0
    f_max = sr / 2
    m_min = 2595.0 * np.log10(1.0 + f_min / 700.0)
    m_max = 2595.0 * np.log10(1.0 + f_max / 700.0)
    mel_points = np.linspace(m_min, m_max, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        if center > left:
            filters[i - 1, left:center] = (np.arange(left, center) - left) / max(center - left, 1)
        if right > center:
            filters[i - 1, center:right] = (right - np.arange(center, right)) / max(right - center, 1)
    enorm = 2.0 / np.maximum(hz_points[2 : n_mels + 2] - hz_points[:n_mels], 1e-9)
    filters *= enorm[:, np.newaxis]
    return filters


@lru_cache(maxsize=1)
def hann_window():
    return torch.hann_window(2048)


def mel_spectrogram(y):
    waveform = torch.from_numpy(y.astype(np.float32))
    stft = torch.stft(
        waveform,
        n_fft=2048,
        hop_length=512,
        win_length=2048,
        window=hann_window(),
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )
    power = (stft.abs() ** 2).numpy()
    return np.maximum(np.matmul(mel_basis(), power), 1e-12)


def power_to_db(mel, ref=1.0, top_db=80):
    db = 10.0 * np.log10(np.maximum(mel, 1e-12))
    db -= 10.0 * np.log10(max(ref, 1e-12))
    db = np.maximum(db, db.max() - top_db)
    return db


def shift_spec(spec, shift):
    if shift == 0:
        return spec
    output = torch.zeros_like(spec)
    if shift > 0:
        output[..., shift:] = spec[..., :-shift]
    else:
        output[..., :shift] = spec[..., -shift:]
    return output


def segment_features(y, sr):
    rms = float(np.sqrt(np.mean(y**2))) if y.size else 0.0
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if y.size:
        zcr = float(np.mean(np.abs(np.diff(np.signbit(y)))))
        window = np.hanning(len(y)).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(y * window)) + 1e-12
        freqs = np.fft.rfftfreq(len(y), d=1.0 / sr)
        total = float(np.sum(spectrum))
        centroid = float(np.sum(freqs * spectrum) / max(total, 1e-12))
        flatness = float(np.exp(np.mean(np.log(spectrum))) / max(np.mean(spectrum), 1e-12))
        cumulative = np.cumsum(spectrum)
        rolloff = float(freqs[min(np.searchsorted(cumulative, 0.85 * cumulative[-1]), len(freqs) - 1)])
    else:
        zcr = 0.0
        centroid = 0.0
        flatness = 0.0
        rolloff = 0.0
    return {
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "zero_crossing_rate": round(zcr, 6),
        "spectral_centroid": round(centroid, 3),
        "spectral_flatness": round(flatness, 6),
        "spectral_rolloff": round(rolloff, 3),
    }


def speaker_embedding(y, sr):
    if y.size < sr // 2:
        y = np.pad(y, (0, sr // 2 - y.size))
    mel = mel_spectrogram(y)
    log_mel = np.log(np.maximum(mel, 1e-12))
    bands = np.array_split(log_mel, 12, axis=0)
    emb = np.concatenate(
        [
            np.array([float(np.mean(band)) for band in bands], dtype=np.float32),
            np.array([float(np.std(band)) for band in bands], dtype=np.float32),
            np.array([float(np.percentile(band, 75) - np.percentile(band, 25)) for band in bands], dtype=np.float32),
        ]
    )
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def cosine_distance(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-9:
        return 1.0
    return float(1.0 - np.dot(a, b) / denom)


def classify_acoustic_context(label, features):
    rms = features.get("rms", 0.0)
    centroid = features.get("spectral_centroid", 0.0)
    if label in {"gunshot", "glass_break", "alarm_signal"}:
        return "high_alert"
    if label == "baby_cry":
        return "distress_signal"
    if label == "human_voice":
        if rms > 0.045 or centroid > 3000:
            return "high_arousal_voice"
        return "speech_context"
    return "ambient_context"


def integrity_screen(y, sr):
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    clipping_ratio = float(np.mean(np.abs(y) >= 0.99)) if y.size else 0.0
    snr = estimate_snr(y, sr)
    rms = float(np.sqrt(np.mean(y**2))) if y.size else 0.0
    duration = len(y) / sr if sr else 0.0
    issues = []
    if clipping_ratio > 0.01:
        issues.append("clipping")
    if snr < 5:
        issues.append("low_snr")
    if rms < CFG["low_energy_rms"]:
        issues.append("very_low_energy")
    if duration < 0.4:
        issues.append("too_short")
    if peak == 0:
        issues.append("silent")
    status = "clean" if not issues else "review"
    return {
        "status": status,
        "issues": issues,
        "snr_db": round(snr, 2),
        "peak": round(peak, 6),
        "rms": round(rms, 6),
        "clipping_ratio": round(clipping_ratio, 6),
    }


def compact_probabilities(labels, probs):
    return [
        {"label": label, "probability": round(float(prob), 6)}
        for label, prob in sorted(zip(labels, probs), key=lambda item: item[1], reverse=True)
    ]


def decision_metadata(probabilities):
    if not probabilities:
        return "unknown", []
    top = probabilities[0]
    second = probabilities[1] if len(probabilities) > 1 else None
    secondary = [
        item for item in probabilities[1:]
        if item["probability"] >= 0.2 or top["probability"] - item["probability"] <= 0.15
    ][:2]
    status = "clear"
    if second and (top["probability"] < 0.45 or top["probability"] - second["probability"] <= 0.15):
        status = "ambiguous"
    return status, secondary


def resample_for_hf(y):
    return resample_audio(y, CFG["sample_rate"], 16000)


def get_hf_audio_classifier(model_id):
    if model_id in HF_AUDIO_CLASSIFIERS:
        return HF_AUDIO_CLASSIFIERS[model_id]
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("audio-classification", model=model_id, device=device)
    HF_AUDIO_CLASSIFIERS[model_id] = classifier
    return classifier


def get_whisper_model(model_id):
    if model_id not in WHISPER_MODELS:
        import whisper
        WHISPER_MODELS[model_id] = whisper.load_model(model_id)
    return WHISPER_MODELS[model_id]


def run_hf_audio_classifier(y, model_id, top_k=5):
    try:
        classifier = get_hf_audio_classifier(model_id)
        audio = resample_for_hf(y)
        output = classifier({"array": audio, "sampling_rate": 16000}, top_k=top_k)
        if isinstance(output, dict):
            output = [output]
        return {
            "status": "ok",
            "model": model_id,
            "predictions": [
                {"label": item.get("label", ""), "score": round(float(item.get("score", 0.0)), 6)}
                for item in output
            ],
        }
    except Exception as exc:
        return {"status": "unavailable", "model": model_id, "reason": str(exc)}


@lru_cache(maxsize=1)
def get_pyannote_pipeline():
    from pyannote.audio import Pipeline
    try:
        return Pipeline.from_pretrained(PYANNOTE_MODEL_ID, token=HF_TOKEN)
    except TypeError:
        return Pipeline.from_pretrained(PYANNOTE_MODEL_ID, use_auth_token=HF_TOKEN)


def pyannote_diarize(audio_path):
    if not HF_TOKEN:
        return [], "missing_hf_token"
    try:
        pipeline = get_pyannote_pipeline()
        result = pipeline(str(audio_path))
        turns = []
        for turn, _, speaker in result.itertracks(yield_label=True):
            turns.append({"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)})
        return turns, "ok"
    except Exception as exc:
        return [], f"unavailable: {exc}"


def assign_pyannote_speakers(segments, turns):
    speakers = set()
    for segment in segments:
        if segment["event_label"] != "human_voice":
            continue
        best_speaker = None
        best_overlap = 0.0
        for turn in turns:
            overlap = max(0.0, min(segment["end"], turn["end"]) - max(segment["start"], turn["start"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]
        if best_speaker:
            segment["speaker"] = best_speaker
            speakers.add(best_speaker)
    return len(speakers)


def reference_embeddings(prototype_dir):
    base = Path(prototype_dir) if prototype_dir else None
    if not base or not base.exists():
        return []
    items = []
    extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    for path in base.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        try:
            y = load_audio(path, CFG["sample_rate"])
            label = path.parent.name if path.parent != base else path.stem
            items.append({"label": label, "path": str(path), "embedding": speaker_embedding(y, CFG["sample_rate"])})
        except Exception:
            continue
    return items


def match_reference_speakers(segments, prototype_dir):
    refs = reference_embeddings(prototype_dir)
    if not prototype_dir:
        return "disabled"
    if not refs:
        return "no_valid_reference_audio"
    matched = 0
    for segment in segments:
        if segment["event_label"] != "human_voice":
            continue
        emb = speaker_embedding(segment["_waveform"], CFG["sample_rate"])
        scored = sorted(
            [(cosine_distance(emb, ref["embedding"]), ref) for ref in refs],
            key=lambda item: item[0],
        )
        if scored:
            distance, ref = scored[0]
            segment["reference_speaker"] = {
                "label": ref["label"],
                "distance": round(float(distance), 6),
                "status": "match" if distance <= 0.24 else "weak_match",
            }
            matched += 1
    return f"matched_{matched}_segments"


class AudioContextPipeline:
    def __init__(self, model_path=None, device=None):
        self.model_path = Path(model_path or available_model_path())
        status = model_status(self.model_path)
        if not status["ready"]:
            raise ModelUnavailableError(status["message"])
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = load_checkpoint(self.model_path)
        self.classes = checkpoint.get("classes", TARGET_CLASSES) if isinstance(checkpoint, dict) else TARGET_CLASSES
        metrics = read_json(METRICS_PATH, {})
        cfg = checkpoint.get("cfg", {}) if isinstance(checkpoint, dict) else {}
        self.temperature = float(checkpoint.get("temperature", cfg.get("temperature", metrics.get("temperature", CFG["temperature"])))) if isinstance(checkpoint, dict) else float(metrics.get("temperature", CFG["temperature"]))
        self.tta_shifts = checkpoint.get("tta_shifts", cfg.get("tta_shifts", metrics.get("tta_shifts", CFG["tta_shifts"]))) if isinstance(checkpoint, dict) else metrics.get("tta_shifts", CFG["tta_shifts"])
        state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
        self.model = ResNet34Spectrogram(self.classes)
        self.model.load_state_dict(clean_state_dict(state), strict=True)
        self.model.to(self.device)
        self.model.eval()
        self.metrics = metrics

    def predict_waveform(self, y):
        spec = audio_to_spec(y).unsqueeze(0).to(self.device)
        prob = None
        with torch.inference_mode():
            for shift in self.tta_shifts:
                logits = self.model(shift_spec(spec, int(shift))) / max(self.temperature, 1e-6)
                current = F.softmax(logits, dim=1)
                prob = current if prob is None else prob + current
        probs = (prob / max(1, len(self.tta_shifts))).squeeze(0).detach().cpu().numpy()
        idx = int(np.argmax(probs))
        probabilities = compact_probabilities(self.classes, probs)
        decision_status, secondary_events = decision_metadata(probabilities)
        return {
            "label": self.classes[idx],
            "confidence": round(float(probs[idx]), 6),
            "probabilities": probabilities,
            "decision_status": decision_status,
            "secondary_events": secondary_events,
        }

    def assign_speakers(self, segments):
        centers = []
        for segment in segments:
            if segment["event_label"] != "human_voice":
                continue
            emb = speaker_embedding(segment["_waveform"], CFG["sample_rate"])
            if not centers:
                centers.append(emb)
                segment["speaker"] = "speaker_1"
                continue
            distances = [cosine_distance(emb, center) for center in centers]
            best = int(np.argmin(distances))
            if distances[best] <= CFG["speaker_threshold"]:
                centers[best] = (centers[best] + emb) / 2
                centers[best] = centers[best] / max(np.linalg.norm(centers[best]), 1e-9)
                segment["speaker"] = f"speaker_{best + 1}"
            else:
                centers.append(emb)
                segment["speaker"] = f"speaker_{len(centers)}"
        return len(centers)

    def transcribe(self, segments, whisper_model):
        voice_segments = [
            segment for segment in segments
            if segment["event_label"] == "human_voice" and (segment["end"] - segment["start"]) >= CFG["speech_min_duration"]
        ]
        if not voice_segments:
            for segment in segments:
                segment["transcript_status"] = "skipped"
            return "no_voice_segments"
        try:
            model = get_whisper_model(whisper_model)
        except Exception as exc:
            for segment in segments:
                segment["transcript_status"] = f"whisper_unavailable: {exc}"
            return "unavailable"
        for segment in segments:
            if segment["event_label"] != "human_voice" or (segment["end"] - segment["start"]) < CFG["speech_min_duration"]:
                segment["transcript_status"] = "skipped"
                continue
            audio = segment.get("_waveform")
            if audio is None:
                segment["transcript_status"] = "missing_audio"
                continue
            try:
                y16 = resample_audio(audio, CFG["sample_rate"], 16000)
                result = model.transcribe(
                    y16,
                    fp16=torch.cuda.is_available(),
                    language=None,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    verbose=False,
                )
                text = result.get("text", "").strip()
                segment["transcript"] = text
                segment["language"] = result.get("language", "?")
                segment["transcript_status"] = "ok" if text else "empty"
            except Exception as exc:
                segment["transcript_status"] = f"failed: {exc}"
        return "done"

    def xai_references(self, labels):
        manifest = read_json(XAI_PATH, [])
        output = []
        for item in manifest:
            label = item.get("class")
            artifact = item.get("artifact")
            if label in labels and artifact:
                output.append(
                    {
                        "class": label,
                        "artifact": artifact,
                        "relative_path": str(Path("..") / "audio_resnet_final_vf" / "xai" / artifact),
                    }
                )
        return output

    def analyze(
        self,
        audio_path,
        transcription=True,
        whisper_model="tiny",
        speaker_grouping=True,
        pyannote_diarization=False,
        hf_emotion=False,
        hf_deepfake=False,
        acoustic_context=True,
        integrity=True,
        xai=True,
        prototype_dir="",
    ):
        started = time.time()
        path = Path(audio_path)
        y = load_audio(path, CFG["sample_rate"])
        segments_raw = segment_audio(y, CFG["sample_rate"])
        segments = []
        for index, raw in enumerate(segments_raw):
            features = segment_features(raw["waveform"], CFG["sample_rate"])
            prediction = self.predict_waveform(raw["waveform"])
            segment = {
                "index": index,
                "start": raw["start"],
                "end": raw["end"],
                "event_label": prediction["label"],
                "event_confidence": prediction["confidence"],
                "decision_status": prediction["decision_status"],
                "secondary_events": prediction["secondary_events"],
                "probabilities": prediction["probabilities"][:6],
                "features": features,
                "low_energy": features["rms"] < CFG["low_energy_rms"],
                "_waveform": raw["waveform"],
            }
            if acoustic_context:
                segment["acoustic_context"] = classify_acoustic_context(segment["event_label"], features)
            segments.append(segment)
        hf_emotion_status = "disabled"
        if hf_emotion:
            hf_emotion_status = "no_voice_segments"
            for segment in segments:
                if segment["event_label"] == "human_voice":
                    segment["hf_emotion"] = run_hf_audio_classifier(segment["_waveform"], EMOTION_MODEL_ID, top_k=5)
                    hf_emotion_status = "done"
        hf_deepfake_status = "disabled"
        if hf_deepfake:
            hf_deepfake_status = "no_voice_segments"
            for segment in segments:
                if segment["event_label"] == "human_voice":
                    segment["hf_deepfake"] = run_hf_audio_classifier(segment["_waveform"], DEEPFAKE_MODEL_ID, top_k=5)
                    hf_deepfake_status = "done"
        transcription_status = "disabled"
        if transcription:
            transcription_status = self.transcribe(segments, whisper_model)
        diarization_status = "disabled"
        speakers = 0
        if pyannote_diarization:
            turns, diarization_status = pyannote_diarize(path)
            if turns:
                speakers = assign_pyannote_speakers(segments, turns)
            elif speaker_grouping:
                speakers = self.assign_speakers(segments)
                diarization_status = f"{diarization_status}; local_fallback"
        elif speaker_grouping:
            speakers = self.assign_speakers(segments)
            diarization_status = "local"
        reference_status = match_reference_speakers(segments, prototype_dir)
        labels = [segment["event_label"] for segment in segments]
        event_counts = {label: labels.count(label) for label in self.classes if label in labels}
        clip_probs = {}
        for segment in segments:
            for item in segment["probabilities"]:
                clip_probs[item["label"]] = clip_probs.get(item["label"], 0.0) + item["probability"]
        clip_probs = {label: value / max(1, len(segments)) for label, value in clip_probs.items()}
        confidence_by_class = {label: round(float(clip_probs.get(label, 0.0)), 6) for label in self.classes}
        top_clip = max(clip_probs.items(), key=lambda item: item[1]) if clip_probs else ("unknown", 0.0)
        for segment in segments:
            segment.pop("_waveform", None)
        result = {
            "file": path.name,
            "file_hash": file_sha256(path),
            "duration_seconds": round(len(y) / CFG["sample_rate"], 3),
            "sample_rate": CFG["sample_rate"],
            "model": {
                "name": "resnet34_final",
                "classes": self.classes,
                "path": str(self.model_path),
                "temperature": round(float(self.temperature), 6),
                "tta_shifts": [int(x) for x in self.tta_shifts],
                "metrics": self.metrics,
            },
            "summary": {
                "top_event": top_clip[0],
                "top_event_probability": round(float(top_clip[1]), 6),
                "event_counts": event_counts,
                "mean_confidence_by_class": confidence_by_class,
                "segments_processed": len(segments),
                "speaker_groups": speakers,
                "transcription_status": transcription_status,
                "diarization_status": diarization_status,
                "hf_emotion_status": hf_emotion_status,
                "hf_deepfake_status": hf_deepfake_status,
                "reference_matching_status": reference_status,
            },
            "integrity": integrity_screen(y, CFG["sample_rate"]) if integrity else {"status": "disabled"},
            "timeline": segments,
            "xai_references": self.xai_references(set(labels)) if xai else [],
            "elapsed_seconds": round(time.time() - started, 3),
        }
        return result


def process_file(audio_path, model_path=None, run_transcription=True, whisper_model="tiny", run_diarization=False, run_emotion=True, run_deepfake=True, run_explanations=True, **kwargs):
    pipeline = AudioContextPipeline(model_path=model_path)
    return pipeline.analyze(
        audio_path,
        transcription=run_transcription,
        whisper_model=whisper_model,
        speaker_grouping=run_diarization,
        pyannote_diarization=bool(kwargs.get("pyannote_diarization", False)),
        hf_emotion=bool(kwargs.get("hf_emotion", False)),
        hf_deepfake=bool(kwargs.get("hf_deepfake", False)),
        acoustic_context=run_emotion,
        integrity=run_deepfake,
        xai=run_explanations,
        prototype_dir=kwargs.get("prototype_dir", ""),
    )
