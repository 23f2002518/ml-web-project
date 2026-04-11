# Messy Mashup

Music genre classification for the IITM Deep Learning & Generative AI project. The project targets the `Messy Mashup` Kaggle challenge and packages the full submission stack: training notebook, W&B logging helpers, report generation, and a Hugging Face Space bundle for live inference.

## Student

- Name: Sachin Kumar Ray
- Roll No: 23F2002518
- Email: 23f2002518@ds.study.iitm.ac.in

## Quick Links

- GitHub repo: <https://github.com/23f2002518/ml-web-project/>
- Kaggle notebook: <https://www.kaggle.com/code/godusssop/dl-23f2002518-notebook-t12026>
- Final Kaggle score reference: `0.85477` from notebook version `V26`
- W&B entity: `23f2002518-dl-genai-project`
- W&B project target: `23f2002518-t12026`
- Hugging Face Space target: `sachin-ray/messy-mashup-23f2002518`

## Project Summary

The challenge is to classify short mixed-music clips into one of 10 genres: `blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, and `rock`.

The final submission workflow uses:

- A pretrained EfficientNet spectrogram classifier for strong offline performance
- A custom CNN with squeeze-and-excitation blocks to satisfy the from-scratch model requirement
- An Audio Spectrogram Transformer as the pretrained transformer baseline
- Test-time augmentation and weighted ensembling for Kaggle inference
- ESC-50 noise mixing and synthetic mashup generation to better match the noisy test distribution

## Repository Layout

```text
.
├── notebooks/
│   └── dl-23f2002518-notebook-t12026.ipynb
├── report/
│   ├── assets/
│   └── generated/
├── scripts/
│   ├── backfill_wandb_from_archive.py
│   ├── error_analysis.py
│   ├── generate_report.py
│   ├── project_pipeline.py
│   ├── run_inference.py
│   └── train_models.py
├── space/
│   ├── app.py
│   ├── README.md
│   ├── artifacts/
│   └── requirements.txt
├── requirements.txt
└── submission_packet.md
```

## Main Assets

- Canonical training notebook: [notebooks/dl-23f2002518-notebook-t12026.ipynb](/home/omegatron/Downloads/Kaggle/DL%20GENAI/Messy%20Mashup/ml-web-project/notebooks/dl-23f2002518-notebook-t12026.ipynb)
- W&B archival logger: [scripts/backfill_wandb_from_archive.py](/home/omegatron/Downloads/Kaggle/DL%20GENAI/Messy%20Mashup/ml-web-project/scripts/backfill_wandb_from_archive.py)
- Shared training/inference module: [scripts/project_pipeline.py](/home/omegatron/Downloads/Kaggle/DL%20GENAI/Messy%20Mashup/ml-web-project/scripts/project_pipeline.py)
- Scripted trainer: [scripts/train_models.py](/home/omegatron/Downloads/Kaggle/DL%20GENAI/Messy%20Mashup/ml-web-project/scripts/train_models.py)
- Scripted inference: [scripts/run_inference.py](/home/omegatron/Downloads/Kaggle/DL%20GENAI/Messy%20Mashup/ml-web-project/scripts/run_inference.py)
- Error-analysis script: [scripts/error_analysis.py](/home/omegatron/Downloads/Kaggle/DL%20GENAI/Messy%20Mashup/ml-web-project/scripts/error_analysis.py)
- Report generator: [scripts/generate_report.py](/home/omegatron/Downloads/Kaggle/DL%20GENAI/Messy%20Mashup/ml-web-project/scripts/generate_report.py)
- Hugging Face Space app: [space/app.py](/home/omegatron/Downloads/Kaggle/DL%20GENAI/Messy%20Mashup/ml-web-project/space/app.py)
- Manual submission answers: [submission_packet.md](/home/omegatron/Downloads/Kaggle/DL%20GENAI/Messy%20Mashup/ml-web-project/submission_packet.md)

## Reproducibility

Install the shared project dependencies with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The notebook expects the Kaggle dataset layout:

```text
/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup/
├── genres_stems/
├── ESC-50-master/audio/
└── test.csv
```

The repository now exposes the same pipeline through scripts as well, so preprocessing, training, inference, and error analysis are no longer notebook-only concerns.

Run the full scripted training workflow with:

```bash
python scripts/train_models.py \
  --base /kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup \
  --output-dir artifacts/training
```

Generate a submission from saved checkpoints with:

```bash
python scripts/run_inference.py \
  --base /kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup \
  --checkpoint-dir artifacts/training \
  --submission-out artifacts/inference/submission.csv
```

Produce validation-side error analysis with:

```bash
python scripts/error_analysis.py \
  --base /kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup \
  --checkpoint-dir artifacts/training \
  --model efficientnet-b0
```

## W&B Logging

The notebook is configured to use:

- Entity: `23f2002518-dl-genai-project`
- Project: `23f2002518-t12026`

Set credentials before running the notebook or the archival logger:

```bash
export WANDB_API_KEY=...
export WANDB_ENTITY=23f2002518-dl-genai-project
export WANDB_PROJECT=23f2002518-t12026
```

To backfill archived metrics from the local Kaggle execution snapshot into W&B:

```bash
python scripts/backfill_wandb_from_archive.py \
  --archive-notebook ../dl-23f2002518-notebook-t12026(1).ipynb \
  --results-zip ../results.zip
```

This recreates the preserved EfficientNet and AST runs from the archived notebook output. The custom CNN run is scaffolded from notebook metadata when no executed scratch-model metrics are present in the local snapshot.

For a fully valid scratch-model comparison in W&B, rerun the custom CNN through [scripts/train_models.py](/home/omegatron/Downloads/Kaggle/DL%20GENAI/Messy%20Mashup/ml-web-project/scripts/train_models.py). The archival script stays honest about the local evidence and does not invent missing custom-CNN metrics.

## Report Generation

The report pipeline now generates a Typst source document plus SVG figure assets, then compiles the final PDF directly with Typst. This avoids the page-layout failures that occurred with the earlier HTML-to-WeasyPrint flow.

Generate the report with:

```bash
python scripts/generate_report.py \
  --archive-notebook ../dl-23f2002518-notebook-t12026(1).ipynb \
  --results-zip ../results.zip \
  --typst-bin /path/to/typst
```

The script writes:

- Typst source in `report/generated/report.typ`
- SVG figure assets in `report/generated/figures/`
- PDF in `report/generated/23f2002518_DG_T12026.pdf`
- A small compatibility notice in `report/generated/report.html`

If `typst` is already on your `PATH`, the `--typst-bin` flag is optional. The recommended compiler source is the official Typst release binary for Linux.

## Hugging Face Space

The ready-to-publish Space bundle lives in [space/](/home/omegatron/Downloads/Kaggle/DL%20GENAI/Messy%20Mashup/ml-web-project/space). It serves a single EfficientNet checkpoint for fast CPU inference and keeps the heavier ensemble in the notebook-only workflow. The live UI now returns:

- Ranked genre confidences through a `Label` output
- The full probability distribution as JSON text
- A compact text summary

```bash
cd space
git init
git remote add origin https://huggingface.co/spaces/sachin-ray/messy-mashup-23f2002518
git add .
git commit -m "Initial Space"
git push origin main
```

## Notes

- The archived local snapshot contains real EfficientNet and AST checkpoints plus a final `submission.csv`.
- The scratch-model code path is preserved in both the notebook and the new script-based pipeline, but its executed metrics were not included in the local Kaggle archive that was available in this workspace.
- The new script layout is meant to strengthen viva readiness: preprocessing, training, inference, and error analysis can now be walked through outside the notebook as well.
- The project is structured so the report and repo stay honest about which results were directly preserved versus which components are prepared for rerun.
