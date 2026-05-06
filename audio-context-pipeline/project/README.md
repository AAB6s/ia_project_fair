# Audio Demo Backend

Local demo for audio event context extraction.

## Setup

```bash
python -m pip install -r requirements.txt
```

Place the final checkpoint in `models/`:

```text
resnet34_final.pt
```

Optional modules use `requirements-optional.txt` and environment keys only when enabled.

## Run

```bash
run_demo.bat
```

## What It Does

- Loads audio and converts it to 32 kHz mono.
- Splits audio into 3-second timeline windows.
- Runs the final ResNet34 spectrogram classifier.
- Returns the top event, confidence, all class probabilities, secondary close events, and acoustic summary.
- Keeps optional logic ready for Whisper, diarization, emotion, synthetic-voice checks, and reference matching.

By default, the demo uses the final CNN and Whisper-ready logic only.
