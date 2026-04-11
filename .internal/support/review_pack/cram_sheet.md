# L1 Master Cram Sheet

This is the one-file last-minute revision sheet.

## 30-Second Project Summary

My project is `Messy Mashup`, a noisy music genre classification system for 10 classes. The main challenge is that the test audio is mixed and noisy, so I used mel-spectrogram preprocessing, synthetic mashup generation, ESC-50 noise injection, three models, and TTA plus weighted ensembling for robust inference. The three models are EfficientNet-B0, a custom CNN with squeeze-and-excitation blocks, and AST. My final Kaggle score reference is `0.85477` from notebook `V26`, and I deployed EfficientNet-B0 on Hugging Face Spaces for lightweight live inference.

## 15-Cell Notebook Map

1. imports and device
2. W&B setup
3. config
4. load metadata
5. preprocessing functions
6. dataset classes
7. model definitions
8. training utilities
9. train EfficientNet
10. train custom CNN
11. train AST
12. compare models
13. run TTA inference
14. weighted ensemble and submission
15. verify output

## Top 20 Short Viva Answers

### Why mel spectrograms?
They convert audio into a compact time-frequency representation that works well with CNNs and transformers.

### Why macro-F1?
Because it values balanced performance across all classes, not just overall accuracy.

### Why `AdamW`?
It is a strong optimizer for deep models and handles weight decay more cleanly than Adam.

### Why `CosineAnnealingLR`?
It reduces the learning rate smoothly across epochs.

### Why label smoothing?
To reduce overconfidence and improve generalization on noisy audio.

### Why `Dropout`?
To reduce overfitting during training.

### Why `BatchNorm2d`?
To stabilize convolutional training.

### Why `optimizer.zero_grad()`?
Because PyTorch accumulates gradients by default.

### Why `torch.no_grad()`?
Because validation and inference do not need gradients.

### Why gradient clipping?
To prevent unstable very large updates.

### Why use EfficientNet?
It is a strong pretrained CNN for spectrogram classification and is also deployment-friendly.

### Why use a custom CNN?
It satisfies the from-scratch requirement and gives an interpretable baseline.

### Why use AST?
It gives a pretrained transformer baseline for comparison.

### Why only one fold for custom CNN?
Because scratch training is slower and the notebook prioritizes practical completion.

### Why TTA?
To reduce sensitivity to exact clip alignment.

### Why weighted ensemble?
Because stronger models should contribute more than weaker ones.

### Why only deploy EfficientNet?
Because it is the strongest lightweight single model for CPU inference.

### Final score?
`0.85477` from notebook `V26`.

### What is preserved in W&B?
EfficientNet and AST archived evidence are preserved locally; the custom-CNN metric history is not preserved locally in the same way.

### How do you explain AI help?
I used AI as an assistant, but I have reviewed the project end to end and can explain the pipeline, model choices, training, inference, and deployment.

## Top 10 Theory Definitions

- `Dataset`: defines how to fetch one sample
- `DataLoader`: batches dataset samples
- `forward`: defines model computation
- `CrossEntropyLoss`: standard multi-class classification loss
- `ReLU`: common non-linear activation
- `Sigmoid`: used here for channel attention weights
- `Conv2d`: learns local 2D patterns
- `Linear`: fully connected layer
- `softmax`: converts logits to probabilities
- `StratifiedKFold`: preserves class balance across folds

## Live Coding Skeletons

### Two-layer NN

```python
class TwoLayerNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
```

### Training Loop Skeleton

```python
for epoch in range(num_epochs):
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
    scheduler.step()
```

### Accuracy Skeleton

```python
def accuracy_fn(logits, y):
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()
```

## Honest Risk Answers

### Custom-CNN archived metrics gap
The custom CNN implementation is fully present and it is my from-scratch model, but the local archived metric history was not preserved the same way as EfficientNet and AST, so I explain that honestly instead of inventing values.

### AI assistance
I used AI heavily as an assistant during implementation, but I have reviewed the full project and can explain the code, training logic, and design choices clearly now.

## Final 5-Minute Revision Checklist

- memorize 3 model names
- memorize final score `0.85477`
- memorize `AdamW`, `CosineAnnealingLR`, macro-F1, label smoothing
- memorize `zero_grad`, `no_grad`, gradient clipping
- memorize why EfficientNet was deployed
- rehearse custom-CNN honest explanation once
