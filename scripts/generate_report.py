from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import textwrap
from collections import Counter
from pathlib import Path
from zipfile import ZipFile


EFFNET_PATTERN = re.compile(r"Ep\s*(\d+): TF1=(\d+\.\d+), VF1=(\d+\.\d+)")
AST_PATTERN = re.compile(r"Ep\s*(\d+): TF1=(\d+\.\d+), VF1=(\d+\.\d+)")

GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]

REPORT_HTML_NOTICE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Messy Mashup Report</title>
  <style>
    body {
      font-family: "Segoe UI", Arial, sans-serif;
      max-width: 760px;
      margin: 48px auto;
      padding: 0 24px;
      line-height: 1.55;
      color: #0f172a;
      background: #f8fafc;
    }
    code {
      background: #e2e8f0;
      padding: 2px 6px;
      border-radius: 6px;
    }
    a {
      color: #2563eb;
    }
  </style>
</head>
<body>
  <h1>Messy Mashup Report</h1>
  <p>
    The canonical report source is now <code>report.typ</code>, and the final PDF is
    <code>23f2002518_DG_T12026.pdf</code>.
  </p>
  <p>
    Open <a href="./23f2002518_DG_T12026.pdf">the generated PDF</a> for the final academic report,
    or inspect <a href="./report.typ">the Typst source</a> for the editable source document.
  </p>
</body>
</html>
"""


def load_notebook(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def cell_text(cell: dict) -> str:
    parts: list[str] = []
    for output in cell.get("outputs", []):
        if "text" in output:
            parts.extend(output["text"])
        elif "data" in output and "text/plain" in output["data"]:
            parts.extend(output["data"]["text/plain"])
    return "".join(parts)


def extract_histories(notebook: dict) -> dict[str, list[tuple[int, float, float]]]:
    histories = {"effnet": [], "ast": []}
    for idx, cell in enumerate(notebook.get("cells", [])):
        text = cell_text(cell)
        if idx == 8:
            histories["effnet"] = [
                (int(match.group(1)), float(match.group(2)), float(match.group(3)))
                for match in EFFNET_PATTERN.finditer(text)
            ]
        if idx == 9:
            histories["ast"] = [
                (int(match.group(1)), float(match.group(2)), float(match.group(3)))
                for match in AST_PATTERN.finditer(text)
            ]
    return histories


def extract_submission_distribution(results_zip: Path) -> Counter:
    with ZipFile(results_zip) as zf:
        rows = zf.read("submission.csv").decode("utf-8").strip().splitlines()[1:]
    counter: Counter = Counter()
    for row in rows:
        _, genre = row.split(",")
        counter[genre] += 1
    return counter


def best_epoch(points: list[tuple[int, float, float]]) -> tuple[int, float]:
    epoch, _, value = max(points, key=lambda item: item[2])
    return epoch, value


def sparkline_svg(
    points: list[tuple[int, float, float]],
    title: str,
    stroke_a: str,
    stroke_b: str,
) -> str:
    width = 920
    height = 280
    padding = 34
    if not points:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
            f'role="img" aria-label="{title}"><rect width="{width}" height="{height}" '
            'rx="18" fill="#f8fafc"/><text x="40" y="48" fill="#0f172a" font-size="22" '
            'font-family="Arial, sans-serif">Metrics unavailable.</text></svg>'
        )

    ys_train = [p[1] for p in points]
    ys_val = [p[2] for p in points]
    all_y = ys_train + ys_val
    min_y = min(all_y)
    max_y = max(all_y)
    y_span = max(max_y - min_y, 1e-6)

    def project(series: list[float]) -> str:
        coords = []
        for idx, value in enumerate(series):
            x = padding + idx * ((width - 2 * padding) / max(len(series) - 1, 1))
            y = height - padding - ((value - min_y) / y_span) * (height - 2 * padding)
            coords.append(f"{x:.1f},{y:.1f}")
        return " ".join(coords)

    return f"""
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{title}">
  <rect x="0" y="0" width="{width}" height="{height}" rx="18" fill="#f8fafc"/>
  <line x1="{padding}" y1="{height-padding}" x2="{width-padding}" y2="{height-padding}" stroke="#cbd5e1" stroke-width="3"/>
  <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height-padding}" stroke="#cbd5e1" stroke-width="3"/>
  <polyline fill="none" stroke="{stroke_a}" stroke-width="5" points="{project(ys_train)}"/>
  <polyline fill="none" stroke="{stroke_b}" stroke-width="5" points="{project(ys_val)}"/>
  <text x="{padding}" y="28" fill="#0f172a" font-size="24" font-family="Arial, sans-serif">{title}</text>
  <text x="{width-240}" y="28" fill="{stroke_a}" font-size="18" font-family="Arial, sans-serif">train F1</text>
  <text x="{width-120}" y="28" fill="{stroke_b}" font-size="18" font-family="Arial, sans-serif">val F1</text>
</svg>
"""


def architecture_svg() -> str:
    return """
<svg xmlns="http://www.w3.org/2000/svg" width="920" height="180" viewBox="0 0 920 180" role="img" aria-label="Custom CNN architecture">
  <rect width="920" height="180" rx="18" fill="#f8fafc"/>
  <rect x="24" y="60" width="110" height="60" rx="14" fill="#dbeafe"/>
  <rect x="164" y="60" width="122" height="60" rx="14" fill="#bfdbfe"/>
  <rect x="316" y="60" width="122" height="60" rx="14" fill="#93c5fd"/>
  <rect x="468" y="60" width="122" height="60" rx="14" fill="#60a5fa"/>
  <rect x="620" y="60" width="92" height="60" rx="14" fill="#38bdf8"/>
  <rect x="742" y="60" width="154" height="60" rx="14" fill="#22c55e"/>
  <text x="79" y="96" text-anchor="middle" font-size="16">Mel input</text>
  <text x="225" y="96" text-anchor="middle" font-size="16">64 filters + SE</text>
  <text x="377" y="96" text-anchor="middle" font-size="16">128 filters + SE</text>
  <text x="529" y="96" text-anchor="middle" font-size="16">256 filters + SE</text>
  <text x="666" y="96" text-anchor="middle" font-size="16">GAP</text>
  <text x="819" y="96" text-anchor="middle" font-size="16">FC head / 10 classes</text>
  <line x1="134" y1="90" x2="164" y2="90" stroke="#334155" stroke-width="3"/>
  <line x1="286" y1="90" x2="316" y2="90" stroke="#334155" stroke-width="3"/>
  <line x1="438" y1="90" x2="468" y2="90" stroke="#334155" stroke-width="3"/>
  <line x1="590" y1="90" x2="620" y2="90" stroke="#334155" stroke-width="3"/>
  <line x1="712" y1="90" x2="742" y2="90" stroke="#334155" stroke-width="3"/>
</svg>
"""


def pipeline_svg() -> str:
    return """
<svg xmlns="http://www.w3.org/2000/svg" width="920" height="170" viewBox="0 0 920 170" role="img" aria-label="Project pipeline">
  <rect width="920" height="170" rx="18" fill="#fff7ed"/>
  <rect x="20" y="58" width="132" height="54" rx="14" fill="#fed7aa"/>
  <rect x="188" y="58" width="162" height="54" rx="14" fill="#fdba74"/>
  <rect x="386" y="58" width="164" height="54" rx="14" fill="#fb923c"/>
  <rect x="586" y="58" width="136" height="54" rx="14" fill="#f97316"/>
  <rect x="758" y="58" width="142" height="54" rx="14" fill="#ea580c"/>
  <text x="86" y="91" text-anchor="middle" font-size="16">Stem folders</text>
  <text x="269" y="91" text-anchor="middle" font-size="16">Mashup + ESC-50 noise</text>
  <text x="468" y="91" text-anchor="middle" font-size="16">Mel / AST features</text>
  <text x="654" y="91" text-anchor="middle" font-size="16">Three models</text>
  <text x="829" y="91" text-anchor="middle" font-size="16">TTA ensemble</text>
  <line x1="152" y1="85" x2="188" y2="85" stroke="#9a3412" stroke-width="3"/>
  <line x1="350" y1="85" x2="386" y2="85" stroke="#9a3412" stroke-width="3"/>
  <line x1="550" y1="85" x2="586" y2="85" stroke="#9a3412" stroke-width="3"/>
  <line x1="722" y1="85" x2="758" y2="85" stroke="#9a3412" stroke-width="3"/>
</svg>
"""


def write_svg(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n")


def build_typst(histories: dict[str, list[tuple[int, float, float]]], distribution: Counter) -> str:
    effnet_epoch, effnet_best = best_epoch(histories["effnet"])
    ast_epoch, ast_best = best_epoch(histories["ast"])
    total_predictions = sum(distribution.values())

    distribution_rows = "\n".join(
        f"  [{genre.title()}], [{distribution.get(genre, 0)}],"
        for genre in GENRES
    )
    comparison_rows = "\n".join(
        [
            f"  [EfficientNet-B0], [`{effnet_best:.4f}`], [Pretrained CNN], "
            f"[Best preserved archive run at epoch {effnet_epoch}; chosen for live deployment.],",
            "  [Custom CNN + SE], [N/A], [From scratch], "
            "[Code preserved in the notebook, but the local archive does not contain executed metric logs.],",
            f"  [Audio Spectrogram Transformer], [`{ast_best:.4f}`], [Pretrained transformer], "
            f"[Strong complementary archived run at epoch {ast_epoch}; retained for ensemble diversity.],",
        ]
    )

    return textwrap.dedent(
        f"""
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
          fill: (x, y) => if y == 0 {{ rgb("#f5f7fb") }} else {{ white }},
          table.header([*Metric*], [*Value*], [*Evidence*]),
          [Best archived EfficientNet val F1], [`{effnet_best:.4f}`], [Recovered from the executed Kaggle notebook archive and checkpoint bundle.],
          [Best archived AST val F1], [`{ast_best:.4f}`], [Recovered from the same archived Kaggle execution snapshot.],
          [Final Kaggle score], [`0.85477`], [Reference score recorded in notebook version `V26`.],
        )

        #v(0.75em)

        = Abstract

        This project addresses noisy music-genre classification for the IITM Deep Learning & Generative AI course project using the Messy Mashup Kaggle challenge. The final workflow combines synthetic mashup generation, ESC-50 noise injection, mel-spectrogram preprocessing, pretrained deep audio models, and test-time augmentation. The strongest preserved archived validation result comes from the EfficientNet-B0 model with macro-F1 `{effnet_best:.4f}`, while the final leaderboard reference used for submission reporting is `0.85477` from notebook version `V26`. The main conclusion is that robust spectrogram pipelines plus noise-aware augmentation are effective for this task, while deployment should serve a single lightweight model rather than the full ensemble.

        = Introduction

        The challenge is to predict one of ten music genres for each test clip in the Messy Mashup benchmark: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock. Unlike clean single-source genre datasets, the test environment contains mixed and noisy audio, so the project objective was not only to train accurate classifiers but also to reduce train-test mismatch through synthetic mashups, ESC-50 noise mixing, and robust inference-time augmentation. The repository packages the complete submission stack: the canonical Kaggle notebook, Weights & Biases logging helpers, a final report generator, and a Hugging Face Space for live inference.

        = Dataset & Preprocessing

        The training data is organized as genre-wise stem folders, while the held-out test set is provided through the Kaggle competition interface. The local workflow uses a fixed sample rate of `16 kHz`, ten-second clips, and `128` mel bins for spectrogram generation. When clips are shorter than the target length, they are padded; otherwise they are trimmed to a fixed duration for consistent batching. Each waveform is peak-normalized before feature extraction.

        #table(
          columns: (1.5fr, 0.9fr, 2.6fr),
          align: (left, center, left),
          inset: 7pt,
          stroke: rgb("#d7dee8"),
          fill: (x, y) => if y == 0 {{ rgb("#f5f7fb") }} else {{ white }},
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

        EfficientNet-B0 is the strongest preserved archived model in the current workspace and the best candidate for live CPU deployment. The model operates on mel-spectrogram inputs and benefits from pretrained image-model inductive biases that transfer well to spectrogram classification. In the archived notebook run, the best preserved validation macro-F1 is `{effnet_best:.4f}` at epoch `{effnet_epoch}`. Because this model combines strong validation quality with moderate inference cost, it was given the largest weight in the final Kaggle ensemble and selected as the serving model for Hugging Face Spaces.

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

        The Audio Spectrogram Transformer (AST) serves as the pretrained transformer baseline. AST models long-range relationships on spectrogram patches using self-attention, which is useful when genre information is distributed across broader time-frequency patterns rather than localized events. In the preserved local archive, the best AST validation macro-F1 is `{ast_best:.4f}` at epoch `{ast_epoch}`. Although AST was not selected as the live deployment model because of higher compute cost, it contributed architectural diversity to the offline ensemble.

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
          fill: (x, y) => if y == 0 {{ rgb("#f5f7fb") }} else {{ white }},
          table.header([*Model*], [*Best val F1*], [*Type*], [*Remarks*]),
        {comparison_rows}
        )

        #v(0.35em)

        The final Kaggle submission path uses a weighted ensemble with greater emphasis on EfficientNet because it achieved the strongest preserved validation result, while AST contributes diversity. The custom CNN remains essential for viva authenticity and architectural comparison even though its archived execution metrics were not retained locally. The final leaderboard reference used in this report is `0.85477` from notebook version `V26`.

        The genre counts in the bundled `submission.csv` indicate how the final inference pipeline distributed predictions across the evaluation set. This does not prove class correctness, but it is useful for spotting extreme collapse into a small subset of labels.

        #table(
          columns: (1fr, 0.6fr),
          align: (left, center),
          inset: 7pt,
          stroke: rgb("#d7dee8"),
          fill: (x, y) => if y == 0 {{ rgb("#f5f7fb") }} else {{ white }},
          table.header([*Genre*], [*Predictions*]),
        {distribution_rows}
        )

        #v(0.25em)

        Total predictions in the archived submission file: `{total_predictions}`.

        = Experiment Tracking & Reproducibility

        The project is structured around reproducibility rather than single-use notebook execution. The repository preserves the canonical Kaggle notebook, a backfill utility for Weights & Biases logging, and a deterministic report generator that rebuilds the final write-up directly from the archived notebook outputs and the bundled `results.zip`. This approach makes the final report auditable: every metric quoted in the document comes from the preserved local snapshot or is explicitly labeled as documented leaderboard evidence.

        The intended experiment tracking target is the Weights & Biases project `23f2002518-t12026` under the entity `23f2002518-dl-genai-project`. The archived notebook snapshot includes preserved EfficientNet and AST training traces, while the custom CNN remains present as code and architecture in the notebook even though its executed local metrics were not retained in the available archive. This distinction is important for viva authenticity, because it clarifies which evidence is directly reproducible from the current workspace and which components are ready for rerun.

        == Deployment Note

        The Hugging Face Space intentionally serves only the EfficientNet-B0 checkpoint rather than the full offline ensemble. This keeps latency reasonable on CPU hardware while preserving a model that already performed strongly in the archived validation logs. The Space reproduces the same mel-spectrogram preprocessing, returns ranked class probabilities, and acts as a public demonstration layer rather than a leaderboard-optimized inference pipeline.

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
        """
    ).strip() + "\n"


def resolve_typst_binary(explicit: str | None) -> str:
    if explicit:
        return explicit
    binary = shutil.which("typst")
    if not binary:
        raise SystemExit(
            "Typst compiler not found. Install typst or pass --typst-bin /path/to/typst."
        )
    return binary


def compile_typst(typst_bin: str, report_typ: Path, output_pdf: Path) -> None:
    subprocess.run(
        [typst_bin, "compile", str(report_typ), str(output_pdf)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-notebook", type=Path, required=True)
    parser.add_argument("--results-zip", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("report/generated"),
    )
    parser.add_argument(
        "--typst-bin",
        type=str,
        default=None,
        help="Path to the Typst compiler binary. Falls back to `typst` on PATH.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    notebook = load_notebook(args.archive_notebook)
    histories = extract_histories(notebook)
    distribution = extract_submission_distribution(args.results_zip)

    write_svg(
        figures_dir / "effnet_curve.svg",
        sparkline_svg(histories["effnet"], "EfficientNet F1 progression", "#2563eb", "#0f766e"),
    )
    write_svg(
        figures_dir / "ast_curve.svg",
        sparkline_svg(histories["ast"], "AST F1 progression", "#7c3aed", "#ea580c"),
    )
    write_svg(figures_dir / "custom_cnn_architecture.svg", architecture_svg())
    write_svg(figures_dir / "pipeline.svg", pipeline_svg())

    report_typ = args.output_dir / "report.typ"
    report_typ.write_text(build_typst(histories, distribution))

    report_html = args.output_dir / "report.html"
    report_html.write_text(REPORT_HTML_NOTICE)

    output_pdf = args.output_dir / "23f2002518_DG_T12026.pdf"
    typst_bin = resolve_typst_binary(args.typst_bin)
    compile_typst(typst_bin, report_typ, output_pdf)

    print(report_typ)
    print(report_html)
    print(output_pdf)


if __name__ == "__main__":
    main()
