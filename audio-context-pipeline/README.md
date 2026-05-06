# Audio Context Pipeline

This folder contains the audio-side model work: model comparison, final ResNet34 training, saved evaluation artifacts, and the local demo backend.

## Contents

| Path | Purpose |
|---|---|
| `audio_model_comparison_vf.ipynb` | Scratch vs pretrained model comparison notebook. |
| `audio_resnet_final_vf.ipynb` | Final ResNet34 training notebook. |
| `audio_pipeline_vf_documentation.html` | Audio-specific visual documentation. |
| `audio_resnet_final_vf/` | Final metrics, plots, XAI artifacts, and exported checkpoint folder. |
| `project/` | Local audio demo and inference pipeline. |

## Final Model

`resnet34_final` classifies six audio contexts:

```text
gunshot, glass_break, alarm_signal, human_voice, baby_cry, background
```

The demo returns a segment timeline with confidence, full class probabilities, secondary close events, acoustic features, and optional speech/context outputs.
