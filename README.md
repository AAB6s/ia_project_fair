# Dual Context Pipeline

This repository contains two local machine-learning systems built for evidence context extraction:

- `pdf/`: document evidence classification, explainability, and a local legal RAG demo focused on Tunisian violence-against-women triage.
- `audio-context-pipeline/`: audio event classification, timeline extraction, explainability, and an optional speech/context demo.

Open `dual_model_pipeline_overview_vf.html` for the full visual workflow, model tree, metrics, and integration overview.

## Structure

| Path | Purpose |
|---|---|
| `pdf/` | Document model notebooks, reports, demo backend, and exported evaluation artifacts. |
| `audio-context-pipeline/` | Audio model notebooks, reports, demo backend, and exported evaluation artifacts. |
| `dual_model_pipeline_overview_vf.html` | Consolidated project documentation for both systems. |

## Run The Demos

PDF legal demo:

```bash
cd pdf/project
python -m pip install -r requirements.txt
run_ollama.cmd
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Audio context demo:

```bash
cd audio-context-pipeline/project
python -m pip install -r requirements.txt
run_demo.bat
```

Model checkpoints are not stored in git. Place downloaded checkpoints in each demo `models/` folder before running full inference.
