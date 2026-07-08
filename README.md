# Multimodal Evidence Intelligence

A research prototype for extracting useful context from sensitive evidence files. The project combines two independent pipelines: one for legal document analysis and one for audio event understanding.

## What This Project Does

The system is designed to help review evidence faster by turning raw files into structured, explainable output.

| Pipeline | Folder | Purpose |
|---|---|---|
| Document evidence analysis | `pdf/` | Extracts text, layout, visual evidence, OCR, legal context, quality signals, and tamper signals from documents. |
| Audio context analysis | `audio-context-pipeline/` | Detects important audio events from short timeline windows and returns confidence scores with explainability artifacts. |

## Document Pipeline

The document pipeline is a local FastAPI backend for evidence-oriented document review.

It supports:

- PDF, image, DOCX, TXT, JSON, and XLSX inputs.
- Page rendering, OCR, table extraction, layout detection, and visual region analysis.
- Four CNN document models:
  - content classification
  - evidence relevance
  - quality assessment
  - tamper detection
- Local legal retrieval with TF-IDF by default and optional FAISS embeddings.
- Optional Ollama generation for structured feedback.
- RAG evaluation with Precision@K, Recall@K, Hit Rate@K, MRR@K, nDCG@K, and guardrail checks.
- XAI artifacts such as Grad-CAM, LIME-style perturbation, calibration curves, thresholds, and confusion matrices.

Best recorded document model results:

| Model | Task | Macro F1 | AUC |
|---|---:|---:|---:|
| `cnn1_content` | content type | 0.9375 | 0.9963 |
| `cnn2_evidence` | evidence relevance | 0.9481 | 0.9904 |
| `cnn3_quality` | document quality | 0.7250 | 0.8734 |
| `cnn4_tamper` | tamper detection | 0.9384 | 0.9838 |

Run the document demo:

```bash
cd pdf/project
python -m pip install -r requirements.txt
ollama pull llama3.1:8b
run_ollama.cmd
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Model checkpoints should be placed in:

```text
pdf/project/models/
```

Expected files:

```text
cnn1_content.pt
cnn2_evidence.pt
cnn3_quality.pt
cnn4_tamper.pt
```

## Audio Pipeline

The audio pipeline is a local FastAPI and Gradio demo for timeline-based sound event analysis.

It supports:

- 32 kHz mono audio preprocessing.
- 3-second timeline windows.
- ResNet34 spectrogram classification.
- Event probabilities, confidence scores, secondary event candidates, and acoustic summaries.
- Optional Whisper, speaker diarization, emotion, synthetic-voice checks, and reference matching.
- XAI outputs for important sound classes.

Detected classes:

- gunshot
- glass break
- alarm signal
- human voice
- baby cry
- background

Best recorded audio model results:

| Model | Accuracy | Macro F1 | Weighted F1 | AUC | Latency |
|---|---:|---:|---:|---:|---:|
| `resnet34_final` | 0.9698 | 0.9248 | 0.9705 | 0.9972 | 4.2 ms |

Run the audio demo:

```bash
cd audio-context-pipeline/project
python -m pip install -r requirements.txt
run_demo.bat
```

Model checkpoint should be placed in:

```text
audio-context-pipeline/project/models/
```

Expected file:

```text
resnet34_final.pt
```

## Repository Layout

```text
.
|-- pdf/
|   |-- project/        Document backend, RAG logic, OCR/layout pipeline, legal knowledge base
|   |-- results/        Model reports, XAI artifacts, evaluation outputs
|   `-- *.ipynb         Training and documentation notebooks
|
|-- audio-context-pipeline/
|   |-- project/        Audio backend, demo app, inference pipeline
|   |-- audio_resnet_final_vf/  Metrics, XAI outputs, model reports
|   `-- *.ipynb         Training and comparison notebooks
|
`-- README.md
```

## Tech Stack

- Python
- FastAPI
- Gradio
- PyTorch
- Torchvision
- Librosa
- OCR and document extraction tooling
- TF-IDF retrieval
- Optional FAISS embeddings
- Optional Ollama local generation
- Optional Whisper and Pyannote modules
- Grad-CAM and LIME-style explainability

## Project Scope

This project focuses on technical evidence triage and context extraction. It is not a replacement for legal, medical, psychological, or emergency professionals. Its value is in organizing signals, highlighting risks, and making review work easier to inspect and explain.
