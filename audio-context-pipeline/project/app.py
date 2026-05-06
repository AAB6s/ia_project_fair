import json
import os
import tempfile
from pathlib import Path

import gradio as gr
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from pipeline import AudioContextPipeline, ModelUnavailableError, available_model_path, model_status, optional_runtime_status


APP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = APP_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = Path(os.getenv("MODEL_PATH", available_model_path()))
PIPELINE = None


def get_pipeline():
    global PIPELINE
    if PIPELINE is None:
        PIPELINE = AudioContextPipeline(model_path=MODEL_PATH)
    return PIPELINE


def write_result(result):
    stem = Path(result.get("file", "audio")).stem
    output_path = RESULTS_DIR / f"{stem}_analysis.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def format_model_status():
    status = model_status(MODEL_PATH)
    metrics = status.get("metrics") or {}
    rows = [
        f"Model path: `{status['model_path']}`",
        f"Ready: `{'yes' if status['ready'] else 'no'}`",
        f"Status: `{status['message']}`",
    ]
    if metrics:
        rows.extend(
            [
                f"Macro F1: `{metrics.get('macro_f1', 'n/a')}`",
                f"Accuracy: `{metrics.get('accuracy', 'n/a')}`",
                f"AUC: `{metrics.get('auc_macro_ovr', 'n/a')}`",
                f"ECE: `{metrics.get('ece', 'n/a')}`",
            ]
        )
    optional = optional_runtime_status()
    rows.extend(
        [
            f"Whisper: `{'ready' if optional.get('whisper', {}).get('available') else 'not installed'}`",
            f"Hugging Face audio: `{'ready' if optional.get('huggingface_audio', {}).get('available') else 'not installed'}`",
            f"pyannote: `{'ready' if optional.get('pyannote', {}).get('available') else 'optional / token needed'}`",
            f"SpeechBrain: `{'ready' if optional.get('speechbrain', {}).get('available') else 'optional'}`",
        ]
    )
    return "\n".join(f"- {row}" for row in rows)


def format_summary(result):
    summary = result.get("summary", {})
    integrity = result.get("integrity", {})
    model = result.get("model", {})
    counts = summary.get("event_counts", {})
    count_text = ", ".join(f"{label}: {count}" for label, count in counts.items()) or "none"
    lines = [
        f"### Audio Analysis Result",
        f"File: `{result.get('file')}`",
        f"Duration: `{result.get('duration_seconds')}s`",
        f"Top event: `{summary.get('top_event')}` with `{summary.get('top_event_probability')}` clip probability",
        f"Segments processed: `{summary.get('segments_processed')}`",
        f"Event counts: `{count_text}`",
        f"Speaker groups: `{summary.get('speaker_groups')}`",
        f"Transcription: `{summary.get('transcription_status')}`",
        f"Diarization: `{summary.get('diarization_status')}`",
        f"HF emotion: `{summary.get('hf_emotion_status')}`",
        f"HF deepfake: `{summary.get('hf_deepfake_status')}`",
        f"Reference matching: `{summary.get('reference_matching_status')}`",
        f"Integrity screen: `{integrity.get('status')}`",
        f"Model: `{model.get('name')}`",
    ]
    return "\n\n".join(lines)


def event_table(result):
    rows = []
    timeline = result.get("timeline", [])
    for item in timeline:
        rows.append(
            [
                item.get("start"),
                item.get("end"),
                item.get("event_label"),
                item.get("event_confidence"),
                item.get("decision_status", ""),
                ", ".join(f"{event.get('label')}:{event.get('probability')}" for event in item.get("secondary_events", [])),
                item.get("acoustic_context", ""),
                (item.get("hf_emotion", {}).get("predictions") or [{}])[0].get("label", ""),
                (item.get("hf_deepfake", {}).get("predictions") or [{}])[0].get("label", ""),
                item.get("speaker", ""),
                item.get("reference_speaker", {}).get("label", ""),
                item.get("transcript", ""),
                item.get("transcript_status", ""),
            ]
        )
    return rows


def probability_table(result):
    values = result.get("summary", {}).get("mean_confidence_by_class", {})
    return [[label, value] for label, value in sorted(values.items(), key=lambda item: item[1], reverse=True)]


def run_analysis(audio_file, transcription, whisper_model, speaker_grouping, pyannote_diarization, hf_emotion, hf_deepfake, acoustic_context, integrity, xai, prototype_dir):
    if not audio_file:
        raise gr.Error("Upload an audio file first.")
    if Path(audio_file).stat().st_size > 50 * 1024 * 1024:
        raise gr.Error("File is larger than 50 MB. Use a shorter file for the local demo.")
    try:
        result = get_pipeline().analyze(
            audio_file,
            transcription=transcription,
            whisper_model=whisper_model,
            speaker_grouping=speaker_grouping,
            pyannote_diarization=pyannote_diarization,
            hf_emotion=hf_emotion,
            hf_deepfake=hf_deepfake,
            acoustic_context=acoustic_context,
            integrity=integrity,
            xai=xai,
            prototype_dir=prototype_dir,
        )
    except ModelUnavailableError as exc:
        raise gr.Error(str(exc)) from exc
    output_path = write_result(result)
    return format_summary(result), event_table(result), probability_table(result), result, str(output_path)


css = """
:root {
  --bg: #f4efe5;
  --ink: #17201b;
  --muted: #65716a;
  --line: #d6cbbb;
  --card: rgba(255, 252, 244, 0.92);
  --accent: #0f6f59;
  --accent2: #b25f2d;
}
.gradio-container {
  background:
    radial-gradient(circle at top left, rgba(15, 111, 89, 0.16), transparent 30rem),
    radial-gradient(circle at 80% 10%, rgba(178, 95, 45, 0.12), transparent 24rem),
    linear-gradient(135deg, #f6f1e8, #eee7db);
  color: var(--ink);
  font-family: Georgia, 'Times New Roman', serif;
}
.main-card {
  border: 1px solid var(--line);
  border-radius: 24px;
  padding: 1.15rem;
  background: var(--card);
  box-shadow: 0 18px 48px rgba(41, 33, 20, 0.08);
}
button.primary {
  background: var(--accent) !important;
}
"""


with gr.Blocks(title="Audio Context Pipeline") as demo:
    gr.HTML(f"<style>{css}</style>")
    gr.Markdown(
        """
        <h1>Audio Context Pipeline</h1>

        Final ResNet34 audio event model with Whisper enabled by default. Other context modules stay optional for controlled demos.
        """,
        elem_classes=["main-card"],
    )
    with gr.Row():
        with gr.Column(scale=7):
            audio_input = gr.File(label="Audio file", file_types=[".wav", ".mp3", ".flac", ".ogg", ".m4a"], type="filepath")
        with gr.Column(scale=5):
            model_box = gr.Markdown(format_model_status(), label="Model status")
            whisper_model = gr.Dropdown(label="Whisper model", choices=["tiny", "base", "small"], value="tiny")
            prototype_dir = gr.Textbox(label="Reference speaker folder", placeholder="Optional folder with reference audio files")
    with gr.Row():
        transcription = gr.Checkbox(label="Transcription", value=True)
        speaker_grouping = gr.Checkbox(label="Local speaker grouping", value=False)
        pyannote_diarization = gr.Checkbox(label="pyannote diarization", value=False)
        hf_emotion = gr.Checkbox(label="HF speech emotion", value=False)
        hf_deepfake = gr.Checkbox(label="HF deepfake voice", value=False)
    with gr.Row():
        acoustic_context = gr.Checkbox(label="Acoustic context", value=True)
        integrity = gr.Checkbox(label="Audio integrity screen", value=True)
        xai = gr.Checkbox(label="XAI references", value=True)
    run_button = gr.Button("Analyze Audio", variant="primary")
    summary_output = gr.Markdown(label="Summary")
    timeline_output = gr.Dataframe(
        headers=["start", "end", "event", "confidence", "decision_status", "secondary_events", "acoustic_context", "hf_emotion", "hf_deepfake", "speaker", "reference_speaker", "transcript", "transcript_status"],
        label="Timeline",
        wrap=True,
    )
    probability_output = gr.Dataframe(headers=["class", "mean confidence"], label="Class confidence summary")
    json_output = gr.JSON(label="Full JSON")
    file_output = gr.File(label="Download JSON")
    run_button.click(
        fn=run_analysis,
        inputs=[audio_input, transcription, whisper_model, speaker_grouping, pyannote_diarization, hf_emotion, hf_deepfake, acoustic_context, integrity, xai, prototype_dir],
        outputs=[summary_output, timeline_output, probability_output, json_output, file_output],
    )

demo.queue(default_concurrency_limit=1)

api = FastAPI(title="Audio Context Pipeline API")
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@api.get("/")
async def root():
    return RedirectResponse(url="/gradio")


@api.get("/health")
async def health():
    return {"ok": True, "model": model_status(MODEL_PATH), "optional_runtime": optional_runtime_status()}


@api.get("/models")
async def models_endpoint():
    status = model_status(MODEL_PATH)
    status["optional_runtime"] = optional_runtime_status()
    return status


@api.post("/analyze")
async def analyze_endpoint(
    file: UploadFile = File(...),
    transcription: bool = Form(True),
    whisper_model: str = Form("tiny"),
    speaker_grouping: bool = Form(False),
    pyannote_diarization: bool = Form(False),
    hf_emotion: bool = Form(False),
    hf_deepfake: bool = Form(False),
    acoustic_context: bool = Form(True),
    integrity: bool = Form(True),
    xai: bool = Form(True),
    prototype_dir: str = Form(""),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name.")
    suffix = Path(file.filename).suffix or ".wav"
    temp_path = None
    try:
        payload = await file.read()
        if len(payload) > 50 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File is larger than 50 MB.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=RESULTS_DIR) as handle:
            handle.write(payload)
            temp_path = Path(handle.name)
        result = get_pipeline().analyze(
            str(temp_path),
            transcription=transcription,
            whisper_model=whisper_model,
            speaker_grouping=speaker_grouping,
            pyannote_diarization=pyannote_diarization,
            hf_emotion=hf_emotion,
            hf_deepfake=hf_deepfake,
            acoustic_context=acoustic_context,
            integrity=integrity,
            xai=xai,
            prototype_dir=prototype_dir,
        )
        result["file"] = file.filename
        write_result(result)
        return result
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        await file.close()
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)


app = gr.mount_gradio_app(api, demo, path="/gradio")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
