# Notebook Walkthrough

Use this while opening [../../../notebooks/dl-23f2002518-notebook-t12026.ipynb](../../../notebooks/dl-23f2002518-notebook-t12026.ipynb). This file is meant to be spoken, not memorized word-for-word.

## Opening Script

I will explain the notebook in the natural pipeline order: setup, experiment tracking, configuration, data loading, preprocessing, datasets, model definitions, training utilities, model training, comparison, inference, ensembling, and final verification. The main goal of the project is robust genre classification for noisy mixed audio, so many choices in the notebook are about reducing train-test mismatch rather than just training one model.

## Cell 1: Setup And Imports

`What it does`
This cell installs and imports the project dependencies, sets the torch device, and loads the key libraries used throughout the notebook.

`Why it exists`
It creates one clean starting point so every later cell has the same environment and imports available.

`Exact viva answer`
This cell prepares the environment for the entire notebook. I import PyTorch for modeling and training, librosa for audio preprocessing, timm for EfficientNet, Hugging Face transformers for AST, sklearn for metrics and stratified folds, and W&B for experiment tracking. I also detect whether the notebook is running on GPU or CPU so the training code can move tensors to the right device.

`Likely examiner questions`

### Why use `torch.nn.functional` when you already import `torch.nn`?

`Short answer`
`torch.nn` is used for defining layers and modules, while `torch.nn.functional` is useful for operation-level functions like `softmax`.

`Deeper follow-up answer`
In this notebook, layers such as `Linear`, `Conv2d`, and `Dropout` are defined through `nn`, but I use `F.softmax` during inference because I only need the function there, not a separate module.

`Danger points`
- Do not say `functional` is only for training
- Do not confuse modules with plain functions

### Why detect `cuda`?

`Short answer`
Because tensor operations must run on the same device as the model.

`Deeper follow-up answer`
If a GPU is available, training is much faster, so I use it. If not, the notebook still works on CPU. The device object is then reused throughout training and inference to keep models and tensors aligned.

`Transition`
After setting up the environment, I initialize experiment tracking so the training runs can be recorded cleanly.

## Cell 2: W&B Initialization

`What it does`
This cell reads W&B credentials from environment variables and logs in when an API key is present.

`Why it exists`
It keeps the notebook reusable both with and without experiment tracking.

`Exact viva answer`
This cell makes experiment tracking configurable. If `WANDB_API_KEY` is available, the notebook logs runs to the configured W&B entity and project. If the key is missing, training can still continue locally without breaking the notebook.

`Likely examiner questions`

### Why use environment variables instead of hardcoding the key?

`Short answer`
Hardcoding secrets is unsafe and makes notebooks less portable.

`Deeper follow-up answer`
Environment variables are the safer standard because they separate credentials from source code. They also make the same notebook easier to reuse across Kaggle, Colab, or local execution.

### What do you log later?

`Short answer`
I log training and validation metrics for each model and also a comparison summary.

`Deeper follow-up answer`
The notebook creates separate runs for EfficientNet, the custom CNN, AST, and a comparison run. The main logged signals are losses, macro-F1, accuracy, and best validation summaries.

`Danger points`
- Do not say every local metric history is preserved equally
- Remember the custom-CNN archived evidence is incomplete locally

`Transition`
Once tracking is configured, I define the full experiment setup in one place.

## Cell 3: Configuration

`What it does`
This cell defines paths, audio parameters, training hyperparameters, augmentation settings, TTA shifts, and the list of genres.

`Why it exists`
It makes the notebook easier to understand and tune because all important constants are centralized.

`Exact viva answer`
This is the central configuration cell. It defines dataset locations, sample rate, clip duration, mel-spectrogram settings, number of folds, epochs, batch sizes, learning rates, augmentation probabilities, and the list of 10 target genres. I also set the random seed here so the run is more reproducible.

`Likely examiner questions`

### Why `16 kHz` and `10` seconds?

`Short answer`
They are practical choices that balance audio detail, memory cost, and consistent batching.

`Deeper follow-up answer`
`16 kHz` is a common audio rate that keeps enough information for genre classification while controlling computation. Fixing clips to `10` seconds ensures the models receive uniform-sized inputs, which simplifies batching and training.

### What is `LabelEncoder` doing?

`Short answer`
It maps string genre names to numeric class labels.

`Deeper follow-up answer`
The model trains on numeric class IDs, not strings. Later, during submission generation, I use the inverse mapping to convert predictions back to genre names.

### Why `StratifiedKFold` later?

`Short answer`
Because it preserves class distribution across folds.

`Deeper follow-up answer`
Genre balance matters during validation. Stratification reduces the chance that a fold becomes biased toward certain genres, so model comparison is more reliable.

`Transition`
After configuration, I scan the dataset and build the metadata used by the rest of the notebook.

## Cell 4: Load Data

`What it does`
This cell builds `train_df`, maps labels, loads `test.csv`, and collects ESC-50 noise files.

`Why it exists`
It converts the folder structure into structured metadata for splitting and training.

`Exact viva answer`
In this cell, I scan each genre folder, collect song paths, build a training dataframe, assign numeric labels with `LabelEncoder`, load the Kaggle test metadata, and collect the available ESC-50 noise files. This gives me the structured metadata backbone for preprocessing, training, and inference.

`Likely examiner questions`

### Why use a dataframe for training metadata?

`Short answer`
Because it makes splitting, filtering, and fold assignment easier.

`Deeper follow-up answer`
With a dataframe, I can easily access paths, genres, labels, and later pass train and validation subsets into the dataset classes. It also makes `StratifiedKFold` usage straightforward.

### Why use ESC-50 noise?

`Short answer`
To make training examples more similar to noisy evaluation clips.

`Deeper follow-up answer`
The challenge test data is messy and mixed, so external environmental noise is added to reduce train-test mismatch and improve robustness.

`Danger points`
- Do not say ESC-50 is part of the model
- It is an augmentation source, not a classifier

`Transition`
Once the metadata is ready, I define the audio-processing functions that turn raw waveforms into model-ready features.

## Cell 5: Audio Processing

`What it does`
This cell defines `load_audio`, `load_stems`, `create_mashup`, `add_noise`, and `to_mel`.

`Why it exists`
It is the core preprocessing and augmentation layer of the project.

`Exact viva answer`
This cell is the most important preprocessing block in the notebook. I load waveforms at a fixed sample rate and duration, normalize them, load and combine stems, create synthetic mashups from songs of the same genre, inject ESC-50 noise with random signal-to-noise ratios, and convert the result into normalized log-mel spectrograms. The overall purpose is not just feature extraction, but also to bring the training distribution closer to the noisy evaluation setting.

`Likely examiner questions`

### What does `load_audio` do?

`Short answer`
It loads a waveform, trims or pads it to a fixed length, and normalizes it.

`Deeper follow-up answer`
`load_audio` ensures every clip has a consistent size. That matters because the models expect fixed-size inputs. It also scales the waveform by peak amplitude so the magnitude range stays stable.

### Why create synthetic mashups?

`Short answer`
To simulate the mixed audio conditions of the competition.

`Deeper follow-up answer`
The training data is stem-based and cleaner than the test set. Mashups artificially create harder mixed examples so the model sees more realistic input conditions during training.

### What is SNR?

`Short answer`
Signal-to-noise ratio measures how strong the signal is relative to the added noise.

`Deeper follow-up answer`
A lower SNR means noisier audio. By sampling random SNR values, the notebook exposes the model to different levels of corruption, which improves robustness.

### Why mel spectrograms?

`Short answer`
They are a compact time-frequency representation that works well with deep models.

`Deeper follow-up answer`
Mel scaling is aligned more closely with human auditory perception than raw linear frequency bins, and the 2D structure makes spectrograms suitable for CNNs and transformer-based models.

`Danger points`
- Do not say preprocessing is only normalization
- Do not forget to mention mashups and noise injection

`Transition`
After defining preprocessing functions, I wrap them into dataset classes so PyTorch can feed the models batch by batch.

## Cell 6: Dataset Classes

`What it does`
This cell defines separate training, validation, and test datasets for mel models and AST.

`Why it exists`
Different stages and model families need slightly different input handling.

`Exact viva answer`
This cell turns the preprocessing logic into PyTorch dataset classes. The mel-based models use one path, while AST uses a Hugging Face feature extractor. Training datasets apply augmentation, but validation and test datasets are more deterministic so evaluation remains stable and meaningful.

`Likely examiner questions`

### What are `__len__` and `__getitem__`?

`Short answer`
`__len__` returns dataset size, and `__getitem__` returns one sample.

`Deeper follow-up answer`
PyTorch uses these two methods to index samples and build batches through a `DataLoader`. `__getitem__` is where each training example is actually constructed.

### Why is augmentation only in training?

`Short answer`
Because training should learn from varied data, but validation should measure performance more consistently.

`Deeper follow-up answer`
If heavy augmentation were also applied in validation, the measured performance could become noisier and less comparable. So the notebook keeps validation more stable.

### Why does AST use a feature extractor?

`Short answer`
Because the AST model expects inputs in a Hugging Face-defined format.

`Deeper follow-up answer`
The feature extractor prepares the raw audio into the representation and tensor shape expected by the pretrained AST model, just like tokenization in NLP pipelines.

`Danger points`
- Do not say the dataset stores all tensors permanently in memory
- It builds samples on demand

`Transition`
With the data pipeline ready, I define the three model families used in the project.

## Cell 7: Model Definitions

`What it does`
This cell defines the EfficientNet classifier, SE block, convolution block, custom CNN, and AST wrapper.

`Why it exists`
It collects all model architectures in one place before training begins.

`Exact viva answer`
This cell defines the three model families in the project. EfficientNet-B0 is the pretrained CNN adapted for single-channel mel spectrograms. The custom CNN is built fully from scratch and includes squeeze-and-excitation attention. AST is wrapped so that its logits can be used inside the same training loop style as the CNN models.

`Likely examiner questions`

### What does `forward` do?

`Short answer`
It defines how input flows through the model.

`Deeper follow-up answer`
Every `nn.Module` implements `forward`, and when I call `model(x)`, PyTorch internally uses that method to compute the output.

### What is `SEBlock`?

`Short answer`
It is channel attention using squeeze-and-excitation.

`Deeper follow-up answer`
The block compresses each channel into a global descriptor, learns importance weights with a small MLP, and uses sigmoid outputs to rescale the channels. This helps the model emphasize more useful feature channels.

### Why can EfficientNet work on spectrograms?

`Short answer`
Because spectrograms are 2D structured inputs with local patterns that CNNs can learn well.

`Deeper follow-up answer`
Even though spectrograms are not natural images, they still have local time-frequency patterns. Convolutional backbones can exploit those patterns effectively.

`Danger points`
- Do not confuse channel attention with spatial attention
- Do not say AST is from scratch

`Transition`
After defining the architectures, I define the reusable training, validation, and probability-prediction utilities.

## Cell 8: Training Utilities

`What it does`
This cell defines `train_epoch`, `valid_epoch`, and `predict_proba`.

`Why it exists`
It avoids repeated logic across the three model families.

`Exact viva answer`
This cell standardizes the training loop, validation loop, and probability prediction logic across all models. It collects losses, predictions, and labels, computes macro-F1 and accuracy, and uses softmax when probabilities are needed for inference and ensembling.

`Likely examiner questions`

### Why `optimizer.zero_grad()`?

`Short answer`
Because PyTorch accumulates gradients by default.

`Deeper follow-up answer`
If I skip `zero_grad()`, gradients from previous batches remain and get added to the next batch’s gradients, which would corrupt the update.

### Why `loss.backward()`?

`Short answer`
It computes gradients of the loss with respect to model parameters.

`Deeper follow-up answer`
Backpropagation uses the computation graph built during the forward pass to populate gradients, which the optimizer then uses to update the weights.

### Why gradient clipping?

`Short answer`
To stabilize training by preventing gradients from becoming too large.

`Deeper follow-up answer`
Large gradient norms can make training unstable. Clipping is a simple safeguard against exploding gradients.

### Why `torch.no_grad()` in validation?

`Short answer`
Because validation and inference do not need backpropagation.

`Deeper follow-up answer`
Disabling gradient computation saves memory and compute and makes evaluation faster.

`Danger points`
- Do not say `zero_grad` resets weights
- Do not say `no_grad` changes model predictions conceptually

`Transition`
Once the shared training utilities are ready, I train each model family separately.

## Cell 9: Train EfficientNet-B0

`What it does`
This cell trains EfficientNet-B0 using stratified 5-fold validation, AdamW, cosine annealing, label smoothing, and early stopping logic.

`Why it exists`
This is the strongest preserved archived model in the project and the best deployment candidate.

`Exact viva answer`
In this cell I train EfficientNet-B0 on mel spectrograms with stratified 5-fold cross-validation. I use AdamW for optimization, cross-entropy with label smoothing to reduce overconfidence, cosine annealing for the learning-rate schedule, and macro-F1 as the main validation signal. The best checkpoint from each fold is saved and later reused during ensemble inference.

`Likely examiner questions`

### Why `AdamW`?

`Short answer`
It is a strong optimizer for deep learning and pretrained models.

`Deeper follow-up answer`
AdamW is similar to Adam but handles weight decay more cleanly by decoupling it from the gradient update. That usually gives better regularization behavior.

### Why `CosineAnnealingLR`?

`Short answer`
It reduces the learning rate smoothly over epochs.

`Deeper follow-up answer`
Compared with a step scheduler, cosine annealing changes the learning rate gradually, which can make training more stable.

### Why label smoothing?

`Short answer`
To reduce overconfidence and improve generalization on noisy data.

`Deeper follow-up answer`
The task is ambiguous and noisy, so strictly one-hot training targets can make the model overly certain. Label smoothing softens that effect.

### Why macro-F1 instead of only accuracy?

`Short answer`
Because balanced class performance matters more than raw overall correctness alone.

`Deeper follow-up answer`
Macro-F1 averages class-wise F1 scores equally, so it reflects whether performance is consistently good across all genres.

`Danger points`
- Do not mix Kaggle leaderboard score with fold validation F1

`Transition`
After training the pretrained CNN, I train the from-scratch baseline.

## Cell 10: Train Custom CNN

`What it does`
This cell trains the custom CNN for one fold.

`Why it exists`
It satisfies the requirement for a genuine model built from scratch and gives a baseline against pretrained alternatives.

`Exact viva answer`
This cell trains the from-scratch custom CNN using the same spectrogram pipeline and the same overall training structure as the other models. I use only one fold here because scratch training is slower and more expensive, and the main goal is to include a true custom architecture for comparison and viva discussion.

`Likely examiner questions`

### Why only one fold?

`Short answer`
Because the scratch model is heavier to train from nothing and this notebook prioritizes practical completion.

`Deeper follow-up answer`
The pretrained models were the stronger competition models, so the custom CNN is kept as a real from-scratch baseline rather than the most heavily tuned model in the notebook.

### Why is the scratch model lower-weighted later?

`Short answer`
Because its validation performance is weaker than EfficientNet and AST.

`Deeper follow-up answer`
The final ensemble weights reflect relative validation quality, so lower-performing models contribute less.

### How is this different from EfficientNet?

`Short answer`
EfficientNet uses pretrained image features, while this CNN learns all features from scratch.

`Deeper follow-up answer`
The custom CNN is fully manually designed and trained from random initialization, so it is more interpretable but usually less powerful than a strong pretrained backbone on limited data.

`Danger points`
- Never invent a preserved archived scratch-model metric that does not exist locally

`Transition`
After the from-scratch baseline, I train the transformer-based audio model.

## Cell 11: Train AST

`What it does`
This cell loads AST, prepares the feature extractor, freezes the embedding block, and trains the model.

`Why it exists`
It gives the pretrained transformer baseline required by the project and lets me compare CNNs vs transformers.

`Exact viva answer`
In this cell I fine-tune the Audio Spectrogram Transformer using Hugging Face components. The feature extractor prepares the audio in the format AST expects, the classification model is loaded with the correct number of labels, and I freeze the embedding layer so fine-tuning is more stable and efficient. Then I train the remaining parameters using the same validation strategy and metrics as the other models.

`Likely examiner questions`

### What is a transformer?

`Short answer`
A transformer is a neural architecture that uses self-attention to model relationships between different parts of the input.

`Deeper follow-up answer`
Unlike CNNs, which mainly focus on local receptive fields, transformers can model broader dependencies by letting positions attend to each other.

### Why freeze embeddings?

`Short answer`
To reduce compute and stabilize fine-tuning.

`Deeper follow-up answer`
The pretrained embedding layer already contains useful low-level structure, so freezing it can reduce overfitting and make training more manageable.

### How is AST different from EfficientNet?

`Short answer`
EfficientNet is a CNN, AST is a transformer.

`Deeper follow-up answer`
EfficientNet learns through convolutions on local patterns, while AST uses self-attention over spectrogram patches. AST is usually heavier but offers a different inductive bias.

`Danger points`
- Do not say AST is trained from scratch here

`Transition`
Once all three model families are trained, I compare them before building the final inference pipeline.

## Cell 12: Model Comparison

`What it does`
This cell prints validation F1 summaries and logs the comparison to W&B.

`Why it exists`
It gives a compact side-by-side view of the three model families.

`Exact viva answer`
This cell summarizes the validation macro-F1 performance of EfficientNet, the custom CNN, and AST, and then records a comparison in W&B. The comparison helps justify why certain models get higher ensemble weights later.

`Likely examiner questions`

### Which model performed best?

`Short answer`
EfficientNet is the strongest preserved archived model in the current workspace.

`Deeper follow-up answer`
That is why it became both the highest-weight model in the offline ensemble and the deployment choice for the Hugging Face Space.

### Why still keep the scratch model?

`Short answer`
Because it satisfies the from-scratch requirement and gives a meaningful baseline.

`Deeper follow-up answer`
It is important both academically and practically, because it helps compare handcrafted architecture choices against pretrained alternatives.

`Danger points`
- Do not dismiss the scratch model as useless

`Transition`
After model comparison, I move to test-time inference and augmentation.

## Cell 13: Inference With TTA

`What it does`
This cell generates predictions at multiple temporal offsets for each model.

`Why it exists`
It makes inference more robust to where the informative audio content appears within the clip.

`Exact viva answer`
This cell applies test-time augmentation by shifting the audio window across several offsets and generating predictions for each shifted version. The idea is to reduce sensitivity to exact alignment and get more stable probabilities under noisy conditions.

`Likely examiner questions`

### Why TTA?

`Short answer`
To make predictions less sensitive to one exact crop of the audio.

`Deeper follow-up answer`
If a useful pattern appears slightly earlier or later in the clip, TTA helps the model see multiple aligned views instead of relying on only one offset.

### What are `all_preds` and `all_weights`?

`Short answer`
`all_preds` stores probability outputs and `all_weights` stores the ensemble weights.

`Deeper follow-up answer`
Later, the weighted ensemble combines each stored prediction matrix according to the corresponding stored model-quality weight.

`Danger points`
- Do not say TTA means retraining the model

`Transition`
After collecting augmented predictions, I combine them into the final weighted ensemble.

## Cell 14: Ensemble And Submission

`What it does`
This cell normalizes the weights, combines probabilities, produces final class predictions, and writes `submission.csv`.

`Why it exists`
It turns the many model-shift predictions into one final Kaggle submission file.

`Exact viva answer`
This cell takes all prediction sets from all models and all TTA shifts, normalizes the weights, and computes a weighted probability ensemble. Then it takes the argmax class per row, maps the label IDs back to genre names, sorts by ID, and saves the final `submission.csv` file.

`Likely examiner questions`

### Why weighted ensemble instead of simple average?

`Short answer`
Because some models are stronger than others and should contribute more.

`Deeper follow-up answer`
EfficientNet gets the highest contribution, AST gets a moderate contribution, and the scratch CNN gets a lower contribution because the validation evidence supports that ordering.

### Why inverse-transform labels?

`Short answer`
Because Kaggle submission expects genre names, not encoded integers.

`Deeper follow-up answer`
The model trains on numeric labels for efficiency, but the final competition file must contain the original class names.

`Danger points`
- Do not say the ensemble combines raw logits here
- It combines probabilities after softmax

`Transition`
Finally, I verify the output and summarize the final state of the notebook.

## Cell 15: Verification And Summary

`What it does`
This cell checks that `submission.csv` exists, confirms the row count, prints generated artifacts, and summarizes the three-model pipeline.

`Why it exists`
It is the final sanity check before submission.

`Exact viva answer`
This cell makes sure the output file was actually produced and that its row count matches the expected test size. It also prints the generated files and ends with a concise summary of the three models and the final ensemble, so the notebook closes with a clear verification step rather than stopping abruptly after inference.

`Likely examiner questions`

### How do you know the submission is valid?

`Short answer`
Because I check that the file exists and that the row count matches the expected number of test samples.

`Deeper follow-up answer`
That does not guarantee leaderboard quality, but it does guarantee the submission file is structurally correct and ready for Kaggle.

`Danger points`
- Do not claim local verification alone proves leaderboard quality

## Final Closing Summary

So the whole notebook is structured as: prepare environment, configure the experiment, build an audio preprocessing and augmentation pipeline, define three model families, train and compare them with shared utilities, then use TTA plus weighted ensembling to generate a final Kaggle submission for noisy music genre classification.
