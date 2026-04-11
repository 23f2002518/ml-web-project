# L1 Concepts Guide

This file assumes zero comfort with the theory and turns each concept into something you can explain in the viva.

## 1. `Dataset`

`Beginner explanation`
A `Dataset` tells PyTorch how to fetch one training example at a time.

`Short viva answer`
I use `Dataset` to define how one audio sample is loaded, augmented, converted to a mel spectrogram, and paired with its label.

`Deeper viva answer`
In this project, the input is not a ready-made tensor file. Each example may involve loading stems, creating mashups, adding noise, and converting to mel spectrograms. So a custom `Dataset` is needed to build each sample dynamically.

`Where in notebook`
Cell 6

`Likely counter-question`
What is the difference between `Dataset` and `DataLoader`?

## 2. `DataLoader`

`Beginner explanation`
A `DataLoader` groups dataset samples into batches and helps with iteration.

`Short viva answer`
I use `DataLoader` to batch data, shuffle training samples, and feed mini-batches into the model.

`Deeper viva answer`
The dataset defines sample construction, while the data loader handles batching and ordering. This separation keeps the training loop clean and scalable.

`Where in notebook`
Cells 9, 10, 11, 13

`Likely counter-question`
Why shuffle during training but not validation?

## 3. `forward`

`Beginner explanation`
`forward` defines how input moves through the layers of the model.

`Short viva answer`
In PyTorch, `forward` is the method that defines the model computation from input tensor to output logits.

`Deeper viva answer`
When I write `model(x)`, PyTorch internally calls the model’s `forward` method. That method defines the computation graph used in both training and inference.

`Where in notebook`
Cell 7

`Likely counter-question`
Do you usually call `forward` directly?

## 4. `CrossEntropyLoss`

`Beginner explanation`
This is a standard loss function for multi-class classification.

`Short viva answer`
I use cross-entropy because the task is 10-class genre classification and the model outputs class logits.

`Deeper viva answer`
Cross-entropy compares the predicted class distribution with the true label and penalizes wrong or low-confidence predictions. It is the standard choice for single-label multi-class classification tasks.

`Where in notebook`
Cells 9, 10, 11

`Likely counter-question`
Why not MSE loss?

## 5. Label Smoothing

`Beginner explanation`
Label smoothing makes the target slightly softer instead of completely one-hot.

`Short viva answer`
I use `label_smoothing=0.1` to reduce overconfidence and improve generalization on noisy and ambiguous audio.

`Deeper viva answer`
With noisy data, strict one-hot targets can make the model too certain even when the input is ambiguous. Label smoothing slightly softens the target distribution and often improves robustness.

`Where in notebook`
Cells 9, 10, 11

`Likely counter-question`
Does label smoothing change the number of classes?

## 6. Accuracy

`Beginner explanation`
Accuracy is the fraction of predictions that are exactly correct.

`Short viva answer`
I track accuracy as a secondary metric to monitor general correctness.

`Deeper viva answer`
Accuracy is useful as a sanity check, but it can hide poor performance on some classes. That is why macro-F1 is more important in this project.

`Where in notebook`
Cell 8 and training cells

`Likely counter-question`
Why is accuracy not your main metric?

## 7. Macro-F1

`Beginner explanation`
Macro-F1 computes F1 for each class separately and then averages them equally.

`Short viva answer`
Macro-F1 is my primary metric because it rewards balanced performance across all 10 genres.

`Deeper viva answer`
Unlike plain accuracy, macro-F1 does not let common or easy classes dominate the evaluation. It is a better fit when we care about all classes fairly.

`Where in notebook`
Cell 8 and training cells

`Likely counter-question`
How is F1 related to precision and recall?

## 8. `AdamW`

`Beginner explanation`
AdamW is an optimizer that updates weights using gradient information and adaptive learning rates.

`Short viva answer`
I use AdamW because it is a strong optimizer for deep networks, especially pretrained models and transformer fine-tuning.

`Deeper viva answer`
AdamW is similar to Adam, but it decouples weight decay from the gradient update. That usually gives better regularization behavior and is a common practical choice for modern deep learning training.

`Where in notebook`
Cells 9, 10, 11

`Likely counter-question`
What is the difference between Adam and AdamW?

## 9. `CosineAnnealingLR`

`Beginner explanation`
This scheduler changes the learning rate gradually according to a cosine curve.

`Short viva answer`
I use cosine annealing to reduce the learning rate smoothly over epochs.

`Deeper viva answer`
Compared with abrupt schedulers like StepLR, cosine annealing changes the learning rate more gradually, which often helps with stable convergence and fine-tuning.

`Where in notebook`
Cells 9, 10, 11

`Likely counter-question`
How is it different from StepLR?

## 10. `scheduler.step()`

`Beginner explanation`
This tells the scheduler to update the learning rate.

`Short viva answer`
In this notebook, I call `scheduler.step()` once per epoch after the train and validation pass.

`Deeper viva answer`
The scheduler is epoch-based here, so it should advance after each full epoch rather than after every single batch.

`Where in notebook`
Cells 9, 10, 11

`Likely counter-question`
What happens if you put it inside the batch loop?

## 11. `ReLU`

`Beginner explanation`
ReLU keeps positive values and turns negative values into zero.

`Short viva answer`
I use ReLU because it is a standard efficient activation for CNN and fully connected layers.

`Deeper viva answer`
ReLU adds non-linearity, which is necessary for the network to learn complex patterns. It is simple, fast, and widely used in modern CNNs.

`Where in notebook`
Cell 7

`Likely counter-question`
Why do we need activations at all?

## 12. `Sigmoid`

`Beginner explanation`
Sigmoid maps a value into the range from 0 to 1.

`Short viva answer`
I use sigmoid inside the SE block to generate channel attention weights.

`Deeper viva answer`
The SE block needs learned weights that can scale feature channels smoothly. Sigmoid is a natural choice because it produces bounded attention values between 0 and 1.

`Where in notebook`
Cell 7

`Likely counter-question`
Why not use sigmoid as the final multi-class output?

## 13. `Dropout`

`Beginner explanation`
Dropout randomly turns off some neurons during training.

`Short viva answer`
I use dropout to reduce overfitting in the classifier heads.

`Deeper viva answer`
During training, dropout makes the network less dependent on any one activation pathway, which improves generalization. During inference, dropout is disabled and the full network is used.

`Where in notebook`
Cell 7

`Likely counter-question`
Is dropout active during inference?

## 14. `BatchNorm2d`

`Beginner explanation`
Batch normalization stabilizes feature distributions during training.

`Short viva answer`
I use `BatchNorm2d` in the custom CNN to improve optimization stability.

`Deeper viva answer`
It normalizes intermediate feature maps across the batch, which can make training faster and more stable, especially in deeper convolutional stacks.

`Where in notebook`
Cell 7

`Likely counter-question`
What is the difference between batch norm and dropout?

## 15. `Conv2d`

`Beginner explanation`
A convolution layer learns local patterns on 2D inputs.

`Short viva answer`
I use `Conv2d` because mel spectrograms are 2D time-frequency inputs, and convolutions are good at learning local patterns from them.

`Deeper viva answer`
In a spectrogram, nearby regions often contain meaningful local structure such as harmonics or transients. CNNs are good at extracting those kinds of local features.

`Where in notebook`
Cell 7

`Likely counter-question`
Why can a CNN trained on images still help on spectrograms?

## 16. `Linear`

`Beginner explanation`
A linear layer is a fully connected layer.

`Short viva answer`
I use linear layers in the classification heads and inside the SE block.

`Deeper viva answer`
After feature extraction, the network needs to map learned representations into class logits. Fully connected layers perform that final projection.

`Where in notebook`
Cell 7

`Likely counter-question`
Why not use only convolution all the way to the output?

## 17. `MaxPool2d`

`Beginner explanation`
Pooling reduces feature-map size while preserving strong responses.

`Short viva answer`
I use max pooling to downsample the custom CNN features and reduce spatial size.

`Deeper viva answer`
Pooling helps reduce computation and also increases the receptive field of later layers by compressing the representation step by step.

`Where in notebook`
Cell 7

`Likely counter-question`
Why use max pooling instead of average pooling there?

## 18. `AdaptiveAvgPool2d`

`Beginner explanation`
This layer compresses a feature map to a fixed output size.

`Short viva answer`
I use adaptive average pooling so the final classifier gets a stable feature dimension before the linear head.

`Deeper viva answer`
It avoids hardcoding the final spatial dimensions and gives a compact global summary of each channel before classification.

`Where in notebook`
Cell 7

`Likely counter-question`
Why not flatten the whole feature map directly?

## 19. `softmax`

`Beginner explanation`
Softmax converts logits into probabilities that sum to 1.

`Short viva answer`
I use softmax during inference because the ensemble combines class probabilities, not raw logits.

`Deeper viva answer`
The model outputs logits first. Softmax turns those into interpretable class probabilities, which are then aggregated across TTA shifts and models.

`Where in notebook`
Cell 8 and inference cells

`Likely counter-question`
Why not use argmax directly before ensembling?

## 20. `torch.no_grad()`

`Beginner explanation`
This tells PyTorch not to track gradients.

`Short viva answer`
I use `torch.no_grad()` during validation and inference because gradients are not needed there.

`Deeper viva answer`
It reduces memory usage and speeds up evaluation by avoiding graph construction for backpropagation.

`Where in notebook`
Cell 8

`Likely counter-question`
Does `no_grad` change the model weights?

## 21. `optimizer.zero_grad()`

`Beginner explanation`
This clears old gradients before computing new ones.

`Short viva answer`
I call `zero_grad()` because PyTorch accumulates gradients by default.

`Deeper viva answer`
Without clearing gradients, each batch would add onto the previous gradient values, which would make the parameter updates incorrect for standard mini-batch training.

`Where in notebook`
Cell 8

`Likely counter-question`
Does `zero_grad()` reset the model?

## 22. Gradient Clipping

`Beginner explanation`
Gradient clipping stops gradients from becoming too large.

`Short viva answer`
I use gradient clipping to keep training more stable.

`Deeper viva answer`
When gradients explode, optimization becomes unstable. Clipping is a simple practical way to prevent very large updates from destabilizing training.

`Where in notebook`
Cell 8

`Likely counter-question`
Why clip by norm instead of clipping each value individually?

## 23. `LabelEncoder`

`Beginner explanation`
It converts string class names into integers.

`Short viva answer`
I use `LabelEncoder` so the models can train on numeric targets while still converting predictions back to genre names later.

`Deeper viva answer`
Deep learning models and loss functions expect numeric class IDs. The label encoder provides both the forward mapping for training and the inverse mapping for final submission.

`Where in notebook`
Cells 3 and 4

`Likely counter-question`
Why not keep strings as labels all the way through?

## 24. `StratifiedKFold`

`Beginner explanation`
It splits the data into folds while preserving class balance.

`Short viva answer`
I use stratified folds so each validation split keeps a similar genre distribution.

`Deeper viva answer`
That makes the fold metrics more reliable and reduces bias that could happen if one fold accidentally contains too many or too few samples from a certain genre.

`Where in notebook`
Cells 9, 10, 11

`Likely counter-question`
Why not just use a random split?

## 25. `wandb.init`

`Beginner explanation`
Starts a W&B experiment run.

`Short viva answer`
I use `wandb.init` to create separate runs for each model family and one comparison run.

`Deeper viva answer`
This makes it easier to track metrics per model and compare their behavior later rather than mixing everything into one run.

`Where in notebook`
Cells 2, 9, 10, 11, 12

`Likely counter-question`
What metadata do you pass into `wandb.init`?

## 26. `wandb.log`

`Beginner explanation`
Logs metrics to W&B while the experiment is running.

`Short viva answer`
I use `wandb.log` to record training and validation metrics over time.

`Deeper viva answer`
Logging losses, F1, and accuracy lets me compare runs and inspect how each model behaves during training. It is useful for both analysis and report preparation.

`Where in notebook`
Cells 9, 10, 11, 12

`Likely counter-question`
What exactly is preserved locally right now?

## Honest W&B Note

The local workspace preserves real archived EfficientNet and AST summaries. The custom-CNN code path is present, but the executed archived metric history was not preserved locally in the same way. If asked, explain that honestly and calmly.
