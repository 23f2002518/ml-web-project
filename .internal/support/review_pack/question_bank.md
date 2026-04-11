# L1 QA Bank

Use each answer in layers: short answer first, then expand only if the examiner keeps drilling.

## 1. What is the project in one minute?

`Short answer`
This project classifies noisy music clips into 10 genres for the Messy Mashup Kaggle challenge using mel-spectrogram preprocessing, synthetic mashup generation, ESC-50 noise injection, three model families, W&B tracking, and a lightweight Hugging Face deployment.

`Deeper follow-up`
The main challenge is train-test mismatch. The training data is cleaner and stem-based, while the test data is mixed and noisy. So the pipeline is designed not only around model choice, but also around robust augmentation and inference.

`Danger points`
- Do not say `it is just a normal genre classifier`
- Do not ignore the noisy mixed-audio angle

## 2. What models did you use?

`Short answer`
I used EfficientNet-B0, a custom CNN with squeeze-and-excitation blocks, and an Audio Spectrogram Transformer.

`Deeper follow-up`
EfficientNet is the pretrained CNN, the custom CNN is the from-scratch requirement, and AST is the pretrained transformer baseline.

`Danger points`
- Do not forget to say which one is from scratch
- Do not say the custom CNN is pretrained

## 3. Why three models?

`Short answer`
The project requires one pretrained model, one from-scratch model, and one additional model of my choice.

`Deeper follow-up`
The three-model setup also makes comparison meaningful because I can compare pretrained CNN features, custom CNN inductive bias, and transformer-style attention on the same task.

`Danger points`
- Do not frame it as just a course checkbox

## 4. Why use mel spectrograms?

`Short answer`
They convert audio into a compact time-frequency representation that works well with CNNs and transformers.

`Deeper follow-up`
Mel scaling is more aligned with human auditory perception than raw linear frequency bins, and it gives a 2D representation that CNNs can process effectively.

`Danger points`
- Do not say spectrograms are images in the exact same semantic sense as photos

## 5. Why macro-F1?

`Short answer`
Macro-F1 treats all classes equally, which is better than plain accuracy when balanced genre performance matters.

`Deeper follow-up`
Accuracy can hide poor performance on minority or harder classes, while macro-F1 averages class-wise F1 scores.

`Danger points`
- Do not say macro-F1 is just the same as accuracy

## 6. Why `AdamW`?

`Short answer`
It is a strong optimizer for deep learning and pretrained models because it combines adaptive updates with decoupled weight decay.

`Deeper follow-up`
Compared with Adam, AdamW handles regularization more cleanly because weight decay is decoupled from the gradient-based update.

`Danger points`
- Do not say `W` means warmup

## 7. Why `CosineAnnealingLR`?

`Short answer`
It reduces the learning rate smoothly over training, which helps stable optimization.

`Deeper follow-up`
Compared with StepLR, cosine annealing avoids abrupt drops and often gives smoother convergence.

`Danger points`
- Do not say the scheduler updates gradients directly

## 8. What does `optimizer.zero_grad()` do?

`Short answer`
It clears previously accumulated gradients before the next backward pass.

`Deeper follow-up`
PyTorch accumulates gradients by default, so if I skip `zero_grad`, the gradients from multiple batches would be added together unintentionally.

`Danger points`
- Do not say it resets model weights

## 9. What does `torch.no_grad()` do?

`Short answer`
It disables gradient computation.

`Deeper follow-up`
I use it during validation and inference because I do not need backpropagation there, and it saves memory and compute.

`Danger points`
- Do not say it changes the model architecture

## 10. Why gradient clipping?

`Short answer`
It keeps gradient norms from exploding and improves training stability.

`Deeper follow-up`
Even if the model is not an RNN, clipping can still help when gradients become too large.

`Danger points`
- Do not say it always improves accuracy

## 11. Why use `Dropout`?

`Short answer`
It reduces overfitting by randomly dropping activations during training.

`Deeper follow-up`
At inference time dropout is disabled, so the full network is used.

`Danger points`
- Do not say dropout is active in inference

## 12. Why use `BatchNorm2d`?

`Short answer`
It stabilizes training by normalizing intermediate feature maps.

`Deeper follow-up`
It can improve optimization speed and reduce internal covariate shift in practice.

`Danger points`
- Do not confuse it with layer normalization

## 13. What is `SEBlock`?

`Short answer`
It is a squeeze-and-excitation attention block that learns channel-wise importance weights.

`Deeper follow-up`
It pools each channel globally, passes the result through a small MLP, then uses sigmoid weights to rescale channels.

`Danger points`
- Do not describe it as spatial attention

## 14. Why can EfficientNet work on spectrograms?

`Short answer`
Because spectrograms are structured 2D inputs with local time-frequency patterns that CNNs can learn well.

`Deeper follow-up`
The semantics are different from natural images, but local edges, textures, and patterns still exist in the time-frequency domain.

`Danger points`
- Do not say the model understands music naturally from ImageNet

## 15. What is AST?

`Short answer`
AST stands for Audio Spectrogram Transformer, a transformer-based classifier for audio spectrogram inputs.

`Deeper follow-up`
It uses self-attention over spectrogram patches instead of relying only on local convolutions.

`Danger points`
- Do not say AST works directly on raw waveforms in this notebook

## 16. Why freeze AST embeddings?

`Short answer`
To reduce training cost and make fine-tuning more stable.

`Deeper follow-up`
When data is limited, freezing early components can reduce overfitting and preserve pretrained representations.

`Danger points`
- Do not say the whole model is frozen

## 17. Why only deploy EfficientNet and not the full ensemble?

`Short answer`
EfficientNet gives strong performance with much lighter CPU inference than the full ensemble.

`Deeper follow-up`
The offline ensemble is better for Kaggle scoring, but the Space is meant for stable public inference on CPU, so I deployed the strongest lightweight single model.

`Danger points`
- Do not say the ensemble cannot be deployed at all

## 18. What is W&B doing in this project?

`Short answer`
It tracks model runs, metrics, and comparisons across experiments.

`Deeper follow-up`
I use separate runs for EfficientNet, the custom CNN, AST, and a comparison summary.

`Danger points`
- Do not imply every archived run is fully preserved equally

## 19. What is your final score?

`Short answer`
My final Kaggle score reference is `0.85477` from notebook version `V26`.

`Deeper follow-up`
That is above the `0.80` cutoff and comes from the final Kaggle-facing notebook version.

`Danger points`
- Do not mix leaderboard score with validation F1

## 20. Why is the custom CNN not the strongest model?

`Short answer`
Because pretrained backbones start with richer features, while the custom CNN has to learn everything from this task.

`Deeper follow-up`
The custom model is still important because it provides a true from-scratch baseline and helps compare architectural choices against pretrained alternatives.

`Danger points`
- Do not insult your own scratch model

## 21. How do you explain the custom-CNN archived metric gap?

`Short answer`
The from-scratch code path is fully present in the notebook and repo, but the local archived workspace did not preserve the executed custom-CNN metric history the same way it preserved EfficientNet and AST.

`Deeper follow-up`
So in the local W&B archival evidence, EfficientNet and AST have preserved metrics, while the custom CNN is clearly marked as needing rerun. I explain that honestly instead of inventing numbers.

`Danger points`
- Never invent a missing F1 value
- Never say `I forgot to train it`

## 22. How should you answer if asked about AI help?

`Short answer`
I used AI heavily as an assistant for implementation support and iteration speed, but I have reviewed the project end to end and can explain the pipeline, architecture, training choices, and deployment.

`Deeper follow-up`
I do not hide the AI assistance, but I focus on demonstrating actual understanding now: data flow, model logic, metrics, and code-level reasoning.

`Danger points`
- Do not say `AI did everything and I do not know it`
- Do not claim zero AI use if that is false

## 23. What improvements would you mention for future work?

`Short answer`
I would rerun the scratch model with full preserved metrics, expand error analysis, try stronger validation logging, and explore more robust audio augmentations or better ensemble calibration.

`Deeper follow-up`
I would also compare more audio-native backbones, add richer confusion-matrix based analysis, and tune deployment further if latency budget allows.

`Danger points`
- Do not say `I would completely rewrite everything`

## 24. What should you show in the GitHub commit history?

`Short answer`
I would show that the project evolved over time through modeling, report stabilization, deployment fixes, and then a cleaner script-based pipeline.

`Deeper follow-up`
The easiest commits to talk about are `580517e` for final deliverables, `5f02225` for report and Space recovery, and `0bc8de3` for the script-based training and analysis pipeline. These show packaging, debugging, and structure improvements.

`Danger points`
- Do not say `I do not remember what any commit means`
- Do not click random commits without a story

## 25. What is the repo structure?

`Short answer`
The repo is organized around the canonical notebook, helper scripts, report generation, the Hugging Face Space, and the manual submission packet.

`Deeper follow-up`
The key folders are `notebooks/` for the Kaggle notebook, `scripts/` for report generation and the script-based pipeline, `report/generated/` for the final report outputs, `space/` for deployment, and the internal review pack for preparation notes.

`Danger points`
- Do not list folders mechanically without saying what they are for

## 26. What preprocessing steps did you take?

`Short answer`
I load fixed-length audio, normalize it, mix stems, generate synthetic mashups, inject ESC-50 noise, and convert the result into normalized log-mel spectrograms.

`Deeper follow-up`
The purpose of preprocessing is not just feature extraction. It is also domain adaptation, because the test clips are noisy and mixed while the raw training stems are cleaner.

`Danger points`
- Do not only say `I made spectrograms`
- Do not forget mashups and noise injection

## 27. Why did you use label smoothing?

`Short answer`
I used `label_smoothing=0.1` in cross-entropy to reduce overconfidence and improve generalization on noisy audio.

`Deeper follow-up`
Because the competition data is messy and sometimes ambiguous, softening the targets slightly helps the model avoid becoming too certain about potentially noisy examples.

`Danger points`
- Do not say label smoothing changes the class count

## 28. What is in your report?

`Short answer`
The report covers the problem statement, dataset and preprocessing, the three models, comparative performance, challenges, and future work.

`Deeper follow-up`
More specifically, it includes dataset and augmentation design, EfficientNet, custom CNN, AST, performance comparison, the final Kaggle score reference, the deployment summary, and an honest note about the preserved archival evidence.

`Danger points`
- Do not say `the report only contains results`

## 29. What exactly should you show in W&B?

`Short answer`
I should show the model-specific runs and the comparison summary, especially the preserved EfficientNet and AST metrics.

`Deeper follow-up`
The key preserved local values are EfficientNet best val F1 `0.9603` and AST best val F1 `0.9397`. The custom CNN is clearly marked as needing rerun in the archival comparison, and I should explain that honestly.

`Danger points`
- Do not open W&B and look surprised by the missing scratch-model metric history

## Mock Viva Script

### GitHub Intro

This is my repo for the `Messy Mashup` project. The commit history shows steady progress over time: final packaging, report stabilization, Space fixes, and then a script-based pipeline for training, inference, and error analysis.

### Notebook Intro

This is the canonical notebook. It starts with setup and W&B, then defines configuration, preprocessing, datasets, three model families, shared training utilities, training cells, comparison, TTA, weighted ensemble inference, and final submission verification.

### Concept Drill

I use AdamW because it combines adaptive optimization with decoupled weight decay. I use cosine annealing because it changes the learning rate smoothly across epochs. I use macro-F1 as the main metric because class-balanced performance matters more than raw accuracy alone.

### Live Coding Prompt

If asked to code a small network, I will define an `nn.Module`, add linear or convolutional layers in `__init__`, define the forward pass, then show a minimal training loop with `zero_grad`, `forward`, `loss.backward`, `optimizer.step`, and `scheduler.step`.

### W&B, Report, Deployment

In W&B, the preserved local archive shows real EfficientNet and AST summaries, while the custom CNN is honestly marked as needing rerun. In the report, I explain the dataset, preprocessing, three models, comparative results, challenges, and future work. For deployment, I use EfficientNet-B0 only because it is the best CPU-friendly single model.
