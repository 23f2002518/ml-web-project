# Models Guide

## Model 1: EfficientNet-B0

### What it is

A pretrained CNN backbone originally trained on ImageNet, adapted here for single-channel mel spectrogram input.

### Why it works on spectrograms

A spectrogram is a 2D grid, just like an image. Even though the semantics differ, convolutional backbones can still learn local patterns across time and frequency.

### Why you used it

- Strong pretrained inductive bias
- Good validation performance
- Lighter than AST for deployment
- Works well with spectrogram inputs

### Architecture summary

- `efficientnet_b0` backbone from `timm`
- input channels changed to `1`
- backbone output dimension `1280`
- classifier head:
  - `Dropout(0.3)`
  - `Linear(1280, 512)`
  - `ReLU`
  - `Dropout(0.2)`
  - `Linear(512, 10)`

### Why good for deployment

It has strong preserved validation performance and lower CPU cost than the transformer model, so it is the safest model to expose in a public Hugging Face Space.

## Model 2: Custom CNN With SE Blocks

### What it is

A fully custom CNN built from scratch on top of mel spectrogram inputs.

### Why it counts as from-scratch

The architecture is manually defined in the notebook using PyTorch layers. It does not use a pretrained backbone.

### Exact block idea

Each convolution block is:

- `Conv2d`
- `BatchNorm2d`
- `ReLU`
- `Conv2d`
- `BatchNorm2d`
- `ReLU`
- `SEBlock`
- `MaxPool2d`

The full network stacks:

- `ConvBlock(1, 64)`
- `ConvBlock(64, 128)`
- `ConvBlock(128, 256)`
- `ConvBlock(256, 512)`
- `AdaptiveAvgPool2d(1)`
- classifier head with dropout and linear layers

### What SE does

SE stands for squeeze-and-excitation. It first compresses each channel into a global descriptor, then learns channel-wise importance weights. Those weights rescale the feature maps, so the network can emphasize more useful channels.

### Why attention is useful here

Different channels may respond to different time-frequency patterns. SE lets the model reweight those channels dynamically instead of treating them all equally.

### Why the scratch model got lower ensemble weight

Its validation performance was lower than EfficientNet and AST, so it is still useful as a baseline, but it contributes less to the final weighted ensemble.

## Model 3: Audio Spectrogram Transformer

### What it is

A pretrained transformer-based audio classifier loaded from Hugging Face.

### What transformer means here

Instead of only using local convolutions, it models relationships across spectrogram patches using self-attention.

### What the feature extractor does

`ASTFeatureExtractor` converts raw audio into the exact input format expected by the AST model. This is similar to tokenization for NLP models, but for audio.

### Why embeddings were frozen

Freezing the embedding layer reduces training cost and can stabilize fine-tuning when data is limited.

### Why AST is heavier

Transformers are usually more expensive than compact CNNs, especially for live inference. That is why AST stayed in the offline evaluation pipeline and was not the deployment model.

## Compare And Contrast

### Pretrained vs from-scratch

Pretrained models already start with useful feature representations learned elsewhere. From-scratch models learn everything from this dataset only. Pretrained models usually converge faster and perform better when data is limited.

### CNN vs transformer

CNNs are strong at local pattern extraction and are usually lighter. Transformers use self-attention to model broader relationships, but are often heavier and more computationally expensive.

### EfficientNet vs AST

EfficientNet is the stronger deployment choice because it is lighter and had the strongest preserved archived validation result. AST is valuable because it gives a transformer baseline and can capture broader spectrogram dependencies.

### Best short comparison answer

EfficientNet was my strongest practical model, the custom CNN satisfied the from-scratch requirement and gave an interpretable baseline, and AST gave me the pretrained transformer comparison point.
