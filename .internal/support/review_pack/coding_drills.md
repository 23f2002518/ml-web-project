# Live Coding Drills

The goal is not to type perfect code at full speed. The goal is to show structure, correctness, and calm explanation.

## 1. Simple Two-Layer Neural Network

```python
import torch
import torch.nn as nn

class TwoLayerNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
```

`What to say while coding`
This is a basic feedforward network with one hidden layer. The first linear layer creates a hidden representation, ReLU adds non-linearity, and the last linear layer outputs logits for classification.

`If I get stuck, I should say this`
The important structure is input layer, hidden layer, activation, and output logits. Even if I miss a small syntax detail, that is the intended design.

## 2. Generic Training Loop

```python
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            val_loss = criterion(logits, y)

    scheduler.step()
```

`What to say while coding`
I first switch to training mode, clear old gradients, run the forward pass, compute the loss, backpropagate, and update the parameters. Then I switch to evaluation mode, disable gradients during validation, and step the scheduler once per epoch.

`If I get stuck, I should say this`
The essential order is zero grad, forward, loss, backward, optimizer step, then validation under `no_grad`.

## 3. Classification Loss Function

```python
criterion = nn.CrossEntropyLoss()
```

`What to say while coding`
I use cross-entropy because this is a multi-class classification problem. The model should output logits, and the labels should be integer class IDs.

`Better version if asked`

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

`Why`
That matches your actual notebook and shows awareness of regularization.

## 4. Accuracy Function

```python
def accuracy_fn(logits, y):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == y).sum().item()
    return correct / len(y)
```

`What to say while coding`
I convert logits to predicted class indices using argmax, count how many predictions match the labels, and divide by batch size.

`If I get stuck, I should say this`
Accuracy is simply the proportion of correct predictions.

## 5. Small Custom CNN Block

```python
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)
```

`What to say while coding`
The convolution learns local patterns, batch norm stabilizes training, ReLU adds non-linearity, and max pooling reduces spatial size.

`If I get stuck, I should say this`
This is the core idea used in CNN feature extraction: local filters, normalization, activation, then downsampling.

## 6. Scratch-Model Style SE Block

```python
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weights = self.fc(self.pool(x)).unsqueeze(-1).unsqueeze(-1)
        return x * weights
```

`What to say while coding`
This is channel attention. I squeeze each channel into a compact summary, learn channel importance weights through a small MLP, and rescale the feature maps.

`If I get stuck, I should say this`
The main idea is not the exact syntax. The idea is to learn which channels matter more and amplify or suppress them.

## 7. Minimal Dataset Example

```python
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y
```

`What to say while coding`
`__len__` tells PyTorch how many examples exist, and `__getitem__` returns one example-label pair. A `DataLoader` can then batch and shuffle those samples.

## 8. Explain These Core Lines

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
scheduler.step()
```

`Ready answer`

- `optimizer.zero_grad()` clears previously accumulated gradients
- `loss.backward()` computes gradients through backpropagation
- `optimizer.step()` updates the parameters
- `scheduler.step()` updates the learning rate schedule

## 9. Where `scheduler.step()` Goes

`Correct viva answer`
In this notebook, `scheduler.step()` is called once per epoch after training and validation.

`Why`
Because the scheduler is being used in an epoch-based way, not batch-by-batch.

## 10. If I Freeze During Coding

Say:

I do not remember the exact syntax, but the structure is: define the module, define the forward pass, create the optimizer and criterion, then train with zero grad, forward, loss, backward, and step.
