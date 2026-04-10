#set page(width: 210mm, height: 297mm, margin: (x: 18mm, y: 16mm))
      #set text(size: 10.6pt)
      #set par(
        justify: true,
        leading: 0.8em,
        spacing: 0.5em,
        first-line-indent: 1.35em,
      )
      #set heading(numbering: "1.")
      #show heading.where(level: 1): it => block(above: 1.2em, below: 0.55em)[#it]
      #show heading.where(level: 2): it => block(above: 0.95em, below: 0.35em)[#it]
      #show figure.caption: set text(size: 9.2pt, fill: rgb("#4b5563"))

      #align(center)[#text(18pt, weight: "bold")[Messy Mashup]]
      #align(center)[Sachin Kumar Ray (23F2002518)]
      #align(center)[IITM BS in Data Science and Applications]

      #v(1em)

      #table(
        columns: (2.2fr, 0.8fr, 3fr),
        align: (left, center, left),
        inset: 7pt,
        stroke: rgb("#d7dee8"),
        fill: (x, y) => if y == 0 { rgb("#f5f7fb") } else { white },
        table.header([*Metric*], [*Value*], [*Evidence*]),
        [Best archived EfficientNet val F1], [`0.9603`], [Recovered from the executed Kaggle notebook archive and checkpoint bundle.],
        [Best archived AST val F1], [`0.9397`], [Recovered from the same archived Kaggle execution snapshot.],
        [Final Kaggle score], [`0.85477`], [Reference score recorded in notebook version `V26`.],
      )

      #v(0.75em)

      = Abstract

      This project addresses noisy music-genre classification for the IITM Deep Learning & Generative AI course project using the Messy Mashup Kaggle challenge. The final workflow combines synthetic mashup generation, ESC-50 noise injection, mel-spectrogram preprocessing, pretrained deep audio models, and test-time augmentation. The strongest preserved archived validation result comes from the EfficientNet-B0 model with macro-F1 `0.9603`, while the final leaderboard reference used for submission reporting is `0.85477` from notebook version `V26`. The main conclusion is that robust spectrogram pipelines plus noise-aware augmentation are effective for this task, while deployment should serve a single lightweight model rather than the full ensemble.

      = Introduction

      The challenge is to predict one of ten music genres for each test clip in the Messy Mashup benchmark: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock. Unlike clean single-source genre datasets, the test environment contains mixed and noisy audio, so the project objective was not only to train accurate classifiers but also to reduce train-test mismatch through synthetic mashups, ESC-50 noise mixing, and robust inference-time augmentation. The repository packages the complete submission stack: the canonical Kaggle notebook, Weights & Biases logging helpers, a final report generator, and a Hugging Face Space for live inference.

      = Dataset & Preprocessing

      The training data is organized as genre-wise stem folders, while the held-out test set is provided through the Kaggle competition interface. The local workflow uses a fixed sample rate of `16 kHz`, ten-second clips, and `128` mel bins for spectrogram generation. When clips are shorter than the target length, they are padded; otherwise they are trimmed to a fixed duration for consistent batching. Each waveform is peak-normalized before feature extraction.

      #table(
        columns: (1.5fr, 0.9fr, 2.6fr),
        align: (left, center, left),
        inset: 7pt,
        stroke: rgb("#d7dee8"),
        fill: (x, y) => if y == 0 { rgb("#f5f7fb") } else { white },
        table.header([*Preprocessing step*], [*Value*], [*Purpose*]),
        [Sample rate], [`16 kHz`], [Keeps the audio resolution manageable while preserving strong genre cues.],
        [Clip length], [`10 s`], [Normalizes batch shape across training and inference.],
        [Mel bins], [`128`], [Balances frequency detail and model size for spectrogram learning.],
        [FFT / hop], [`1024 / 256`], [Captures enough temporal detail for rhythm-heavy genres.],
        [Waveform normalization], [Peak], [Reduces loudness variation before spectrogram extraction.],
      )

      #v(0.35em)

      To bridge the gap between cleaner stems and noisier evaluation clips, the pipeline synthesizes mashups from genre stems and injects ESC-50 environmental noise. This step encourages the models to learn robust time-frequency cues instead of overfitting to clean single-track statistics. The final Kaggle inference path also uses test-time augmentation so that predictions are averaged across multiple views of each example rather than relying on a single spectrogram crop.

      #figure(
        image("figures/pipeline.svg", width: 100%),
        caption: [Training and inference pipeline from stem folders to the final TTA ensemble.]
      )

      #v(0.35em)

      = Modeling & Experimentation

      == Model 1: EfficientNet-B0

      EfficientNet-B0 is the strongest preserved archived model in the current workspace and the best candidate for live CPU deployment. The model operates on mel-spectrogram inputs and benefits from pretrained image-model inductive biases that transfer well to spectrogram classification. In the archived notebook run, the best preserved validation macro-F1 is `0.9603` at epoch `26`. Because this model combines strong validation quality with moderate inference cost, it was given the largest weight in the final Kaggle ensemble and selected as the serving model for Hugging Face Spaces.

      #figure(
        image("figures/effnet_curve.svg", width: 100%),
        caption: [EfficientNet-B0 train and validation macro-F1 extracted from the archived Kaggle notebook execution.]
      )

      #v(0.35em)

      == Model 2: Custom CNN With Squeeze-and-Excitation

      The from-scratch baseline is a custom convolutional neural network that stacks mel-spectrogram convolution blocks with squeeze-and-excitation recalibration, followed by global average pooling and a compact fully connected head. This model satisfies the course requirement to include a model built from scratch and acts as an interpretable baseline for comparing handcrafted architecture choices against pretrained alternatives. The local archive retains the code path and design, but not the executed metric logs, so the report explicitly distinguishes between implemented architecture and archived evidence instead of inventing missing numbers.

      #figure(
        image("figures/custom_cnn_architecture.svg", width: 100%),
        caption: [Custom CNN architecture used as the from-scratch baseline in the canonical notebook.]
      )

      #v(0.35em)

      == Model 3: Audio Spectrogram Transformer

      The Audio Spectrogram Transformer (AST) serves as the pretrained transformer baseline. AST models long-range relationships on spectrogram patches using self-attention, which is useful when genre information is distributed across broader time-frequency patterns rather than localized events. In the preserved local archive, the best AST validation macro-F1 is `0.9397` at epoch `17`. Although AST was not selected as the live deployment model because of higher compute cost, it contributed architectural diversity to the offline ensemble.

      #figure(
        image("figures/ast_curve.svg", width: 100%),
        caption: [AST train and validation macro-F1 extracted from the archived Kaggle notebook execution.]
      )

      #v(0.35em)

      = Performance & Comparative Analysis

      Macro-F1 is the primary competition metric and the most relevant evaluation signal because it balances performance across all ten genres instead of letting frequent classes dominate. The table below summarizes the final model set and the evidence preserved in the current local archive.

      #table(
        columns: (1.6fr, 0.9fr, 1fr, 3fr),
        align: (left, center, left, left),
        inset: 7pt,
        stroke: rgb("#d7dee8"),
        fill: (x, y) => if y == 0 { rgb("#f5f7fb") } else { white },
        table.header([*Model*], [*Best val F1*], [*Type*], [*Remarks*]),
        [EfficientNet-B0], [`0.9603`], [Pretrained CNN], [Best preserved archive run at epoch 26; chosen for live deployment.],
[Custom CNN + SE], [N/A], [From scratch], [Code preserved in the notebook, but the local archive does not contain executed metric logs.],
[Audio Spectrogram Transformer], [`0.9397`], [Pretrained transformer], [Strong complementary archived run at epoch 17; retained for ensemble diversity.],
      )

      #v(0.35em)

      The final Kaggle submission path uses a weighted ensemble with greater emphasis on EfficientNet because it achieved the strongest preserved validation result, while AST contributes diversity. The custom CNN remains essential for viva authenticity and architectural comparison even though its archived execution metrics were not retained locally. The final leaderboard reference used in this report is `0.85477` from notebook version `V26`.

      The genre counts in the bundled `submission.csv` indicate how the final inference pipeline distributed predictions across the evaluation set. This does not prove class correctness, but it is useful for spotting extreme collapse into a small subset of labels.

      #table(
        columns: (1fr, 0.6fr),
        align: (left, center),
        inset: 7pt,
        stroke: rgb("#d7dee8"),
        fill: (x, y) => if y == 0 { rgb("#f5f7fb") } else { white },
        table.header([*Genre*], [*Predictions*]),
        [Blues], [281],
[Classical], [333],
[Country], [168],
[Disco], [355],
[Hiphop], [292],
[Jazz], [291],
[Metal], [310],
[Pop], [356],
[Reggae], [277],
[Rock], [357],
      )

      #v(0.25em)

      Total predictions in the archived submission file: `3020`.

      = Experiment Tracking & Reproducibility

      The project is structured around reproducibility rather than single-use notebook execution. The repository preserves the canonical Kaggle notebook, a backfill utility for Weights & Biases logging, and a deterministic report generator that rebuilds the final write-up directly from the archived notebook outputs and the bundled `results.zip`. This approach makes the final report auditable: every metric quoted in the document comes from the preserved local snapshot or is explicitly labeled as documented leaderboard evidence.

      The intended experiment tracking target is the Weights & Biases project `23f2002518-t12026` under the entity `23f2002518-dl-genai-project`. The archived notebook snapshot includes preserved EfficientNet and AST training traces, while the custom CNN remains present as code and architecture in the notebook even though its executed local metrics were not retained in the available archive. This distinction is important for viva authenticity, because it clarifies which evidence is directly reproducible from the current workspace and which components are ready for rerun.

      == Deployment Note

      The Hugging Face Space intentionally serves only the EfficientNet-B0 checkpoint rather than the full offline ensemble. This keeps latency reasonable on CPU hardware while preserving a model that already performed strongly in the archived validation logs. The Space reproduces the same mel-spectrogram preprocessing, returns ranked class probabilities, and acts as a public demonstration layer rather than a leaderboard-optimized inference pipeline.

      #pagebreak()

      = Challenges Faced

      The main technical challenge was mismatch between clean training stems and the noisy, mixed evaluation setting. A second challenge was evidence preservation: the current local archive contains strong EfficientNet and AST runs, but not the executed scratch-model metric history. The third challenge was engineering rather than modeling: the original HTML-to-PDF report layout was visually unstable, and the initial Space app used a Gradio output path that failed in hosted runtime schema generation.

      These issues were handled conservatively. The report generator was rewritten around Typst for page-stable academic output, while the Space UI was simplified to components that are more robust on Hugging Face Spaces. Throughout the repository, the guiding principle is to stay honest about the archive contents rather than fill missing numbers with invented results.

      = Conclusion & Future Work

      The strongest lesson from this project is that matching the competition's noisy evaluation setting matters as much as backbone choice. ESC-50 noise injection, synthetic mashup generation, and TTA all target the real source of difficulty: genre recognition under overlap and contamination. Pretrained spectrogram models proved highly effective, while the from-scratch CNN remains valuable for transparency and viva readiness. The main limitation of the current local archive is missing executed metrics for the custom CNN run, so the report stays explicit about what was preserved versus what could be rerun.

      Future improvements are clear: rerun the scratch model with full W&B logging, calibrate ensemble weights on a stronger validation protocol, explore waveform-native or hybrid models, and add a richer error analysis with confusion matrices and audio exemplars. For deployment, a future revision could add top-k probability plots or confidence-threshold warnings while keeping the live app lightweight enough for CPU hosting.

      = References

      + Tan, M., and Le, Q. V. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML, 2019.
      + Gong, Y., Chung, Y.-A., and Glass, J. AST: Audio Spectrogram Transformer. Interspeech, 2021.
      + Hu, J., Shen, L., and Sun, G. Squeeze-and-Excitation Networks. CVPR, 2018.
      + Piczak, K. J. ESC: Dataset for Environmental Sound Classification. ACM Multimedia, 2015.
      + Kaggle notebook reference: #link("https://www.kaggle.com/code/godusssop/dl-23f2002518-notebook-t12026")[dl-23f2002518-notebook-t12026].
      + GitHub repository: #link("https://github.com/23f2002518/ml-web-project/")[23f2002518/ml-web-project].
      + Hugging Face Space: #link("https://huggingface.co/spaces/sachin-ray/messy-mashup-23f2002518")[sachin-ray/messy-mashup-23f2002518].
