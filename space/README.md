---
title: Messy Mashup 23F2002518
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
python_version: "3.10"
app_file: app.py
pinned: false
---

# Messy Mashup

Live music-genre prediction demo for the IITM DL & GenAI project submission.

## What this Space does

- Accepts a single audio clip upload
- Converts it into a mel spectrogram
- Runs an EfficientNet-based classifier on CPU
- Returns the predicted genre and class confidence distribution

## Model Choice

The Kaggle notebook uses a heavier ensemble for leaderboard inference. This Space intentionally serves a single EfficientNet checkpoint to keep startup time and CPU latency reasonable for a public demo.

## Files

- `app.py`: Gradio app entrypoint
- `artifacts/effnet_f1.pt`: live demo checkpoint
- `requirements.txt`: Space runtime dependencies
