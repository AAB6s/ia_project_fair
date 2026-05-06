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

## Offline RAG Evaluation

```bash
python evaluate_rag.py
```

The runner uses `evaluation/rag_eval_dataset.json` and writes `../results/rag_offline_eval.json` plus `../results/rag_offline_eval.csv`. Add `--use-llm` only when Ollama generation should be included in the benchmark.

## What It Does

- Reads PDF, image, DOCX, TXT, JSON, and XLSX inputs.
- Renders PDF pages, extracts layout regions, text blocks, tables, embedded images, OCR text, and page-level visual previews.
- Runs the four document CNN models on visual evidence regions and page previews.
- Retrieves Tunisian legal context from local storage with TF-IDF by default and optional FAISS sentence embeddings.
- Produces a structured legal triage report with citations, risks, evidence gaps, and next steps.
- Returns run metrics for retrieval, citation grounding, document extraction, and Ollama or fallback generation.
- Blocks unsupported prompts that are outside the document, evidence, OCR/layout, or Tunisian legal-reference scope.
- Includes an offline RAG benchmark with Precision@K, Recall@K, Hit Rate@K, MRR@K, nDCG@K, and guardrail accuracy.

This tool supports review and triage. It is not a substitute for a qualified lawyer.
