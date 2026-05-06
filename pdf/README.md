# Document Evidence Pipeline

This folder contains the document-side model work: final training, lightweight model comparison, saved evaluation artifacts, and the local demo backend.

## Contents

| Path | Purpose |
|---|---|
| `document-processing-pipeline.ipynb` | Final four-model document training notebook. |
| `document_model_comparison_vf.ipynb` | Lightweight comparison notebook for scratch and pretrained candidates. |
| `pipeline_vf_documentation.html` | Document-specific visual documentation. |
| `results/` | Metrics, curves, confusion matrices, calibration, and XAI artifacts. |
| `project/` | Local backend for document analysis, legal retrieval, and LLM feedback. |

## Model Tree

- `cnn1_content`: document/page type classification.
- `cnn2_evidence`: primary vs secondary evidence role.
- `cnn3_quality`: readability and quality assessment.
- `cnn4_tamper`: authenticity/tamper screening.

The backend combines model outputs with OCR/text extraction and a Tunisia-focused legal retrieval layer.
