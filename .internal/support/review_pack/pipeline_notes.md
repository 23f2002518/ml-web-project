# Training And Inference Guide

## End-To-End Story

The whole pipeline is designed around one core problem: the test clips are noisy and mixed, while the raw training stems are cleaner and more separated. So the project tries to reduce train-test mismatch through augmentation, robust validation, and ensemble inference.

## 1. Load Stems

Each training song is stored as four stems:

- drums
- vocals
- bass
- others

`Why`
This gives more control over how to construct training audio than a single already-mixed file would.

## 2. Synthesize Mashups

The notebook can create synthetic mashups by combining stems from different songs of the same genre.

`Why`
The competition test distribution is messy and mixed. Synthetic mashups make training inputs more similar to the evaluation environment.

## 3. Inject ESC-50 Noise

Environmental audio from ESC-50 is mixed into the training audio at random signal-to-noise ratios.

`Why`
This increases robustness to noisy real-world conditions and reduces overfitting to clean stems.

## 4. Convert To Mel Spectrogram

The waveform is converted to a log-mel spectrogram after normalization.

`Why`
Mel spectrograms compress the frequency axis in a way that is meaningful for audio and are a good input format for CNN and transformer models.

## 5. Build Dataset And DataLoader Objects

Training datasets apply augmentation. Validation datasets are more stable and deterministic.

`Why`
Augmentation should help the model learn, but validation should reflect clean evaluation of the trained model.

## 6. Train Each Model

### EfficientNet-B0

- 5 stratified folds
- `AdamW`
- `CrossEntropyLoss(label_smoothing=0.1)`
- `CosineAnnealingLR`

### Custom CNN

- 1 fold in the notebook
- same broad training setup
- used as a from-scratch baseline

### AST

- 5 stratified folds
- Hugging Face feature extractor
- embedding layer frozen

`Why`
This gives one pretrained CNN, one from-scratch model, and one pretrained transformer, matching the project requirement.

## 7. Validate With Macro-F1 And Accuracy

Macro-F1 is treated as the main signal. Accuracy is secondary.

`Why`
Macro-F1 is more appropriate when we care about balanced performance across all 10 genres.

## 8. Save Best Checkpoints

The notebook saves the best checkpoint per fold using validation F1.

`Why`
This avoids keeping only the last epoch if performance has already started degrading.

## 9. Run Test-Time Augmentation

At inference time, the notebook shifts the audio offset across multiple values and predicts each shifted version.

`Why`
This reduces sensitivity to the exact temporal crop and makes the predictions more stable.

## 10. Weighted Ensemble

All prediction sets are combined using normalized weights.

- EfficientNet gets the highest weight
- AST gets a moderate weight
- Custom CNN gets the lowest weight

`Why`
The weights reflect relative validation quality. Better models should contribute more.

## 11. Generate `submission.csv`

The final class IDs are mapped back to genre names and saved in the required Kaggle format.

`Why`
Kaggle expects string labels, not encoded integers.

## Best One-Minute Flow Summary

I start from genre-wise stem folders, create more realistic training examples through mashups and noise injection, convert everything to mel spectrograms, then train three model families: EfficientNet, a custom CNN, and AST. I validate them with macro-F1, save the best checkpoints, and at inference time I apply TTA plus weighted ensembling to generate the final Kaggle submission.
