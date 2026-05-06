# PDF Demo Backend

Local FastAPI backend for document analysis, legal retrieval, and structured feedback.

## Setup

```bash
python -m pip install -r requirements.txt
ollama pull llama3.1:8b
```

Place the four document checkpoints in `models/`:

```text
cnn1_content.pt
cnn2_evidence.pt
cnn3_quality.pt
cnn4_tamper.pt
```

## Run

```bash
run_ollama.cmd
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

## What It Does

- Reads PDF, image, DOCX, TXT, JSON, and XLSX inputs.
- Extracts text and page images.
- Runs the four document CNN models.
- Retrieves Tunisian legal context from local storage.
- Produces a structured legal triage report with citations, risks, evidence gaps, and next steps.

This tool supports review and triage. It is not a substitute for a qualified lawyer.
