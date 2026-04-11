# L1 Rapid Checklist

## Open These Tabs In This Exact Order

1. GitHub repo main page
2. GitHub commit history
3. Canonical notebook
4. W&B project
5. Final report PDF
6. Kaggle notebook and leaderboard score reference
7. Hugging Face deployment

## Best Commits To Click On If Asked

- `580517e` `Finalize project deliverables`
  - Good for explaining the first clean final-project packaging pass.
- `5f02225` `Finalize Typst report and Space recovery`
  - Good for explaining report cleanup and deployment stabilization.
- `0bc8de3` `Add script-based training and analysis pipeline`
  - Good for explaining that the repo now separates training, inference, and error analysis outside the notebook.

## Five-Minute Pre-Viva Setup

- Log in to GitHub, Kaggle, W&B, and Hugging Face.
- Keep your ID ready because many L1 proctors ask for it first.
- Open the notebook at the top and collapse noisy outputs if needed.
- Keep the report PDF at page 1.
- Open the Space once so the UI is already loaded.
- Keep your final score and 3 model names on a sticky note.
- Be ready to type a small PyTorch snippet in a fresh notebook.

## One-Line Answers You Must Know

`What is this project?`
Music genre classification for the Messy Mashup Kaggle challenge using noisy audio augmentation, three model families, W&B tracking, and a lightweight Hugging Face deployment.

`What models did you use?`
I used a pretrained EfficientNet-B0 on mel spectrograms, a custom CNN with squeeze-and-excitation blocks built from scratch, and a pretrained Audio Spectrogram Transformer.

`What is your final score?`
My final Kaggle score reference is `0.85477` from notebook version `V26`.

`What is deployed?`
I deployed the EfficientNet-B0 model on Hugging Face Spaces because it gives strong accuracy with much lighter CPU inference than the full offline ensemble.

## Thirty-Second Opening Script

My project is `Messy Mashup`, a noisy music genre classification system for the IITM DL & GenAI project. The main challenge is that the test clips are mixed and noisy, so I used synthetic mashup generation, ESC-50 noise injection, mel-spectrogram preprocessing, and three models: EfficientNet-B0, a custom CNN from scratch, and AST. I tracked experiments in W&B, used weighted test-time augmentation and ensembling for offline Kaggle inference, achieved a final score reference of `0.85477`, and deployed EfficientNet-B0 on Hugging Face for live CPU-friendly inference.

## If You Panic

- Slow down and explain the flow, not the code line by line.
- If you forget syntax, explain the idea clearly.
- If asked something tough, say:
  - `I do not recall the exact line right now, but the idea is...`
- Never say:
  - `AI did everything`
  - `I do not know anything about this project`
  - `I just copied it`
