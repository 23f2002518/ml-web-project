from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from zipfile import ZipFile


EFFNET_PATTERN = re.compile(r"Ep\s*(\d+): TF1=(\d+\.\d+), VF1=(\d+\.\d+)")
AST_PATTERN = re.compile(r"Ep\s*(\d+): TF1=(\d+\.\d+), VF1=(\d+\.\d+)")
SUBMISSION_PATTERN = re.compile(r"^\s*(\d+)\s+([a-z]+)\s*$", re.MULTILINE)


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


def load_notebook(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


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
                (int(m.group(1)), float(m.group(2)), float(m.group(3)))
                for m in EFFNET_PATTERN.finditer(text)
            ]
        if idx == 9:
            histories["ast"] = [
                (int(m.group(1)), float(m.group(2)), float(m.group(3)))
                for m in AST_PATTERN.finditer(text)
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


def sparkline_svg(
    points: list[tuple[int, float, float]],
    title: str,
    stroke_a: str,
    stroke_b: str,
) -> str:
    width = 460
    height = 170
    padding = 20
    if not points:
        return "<div>Metrics unavailable.</div>"

    xs = [p[0] for p in points]
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
    <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{title}">
      <rect x="0" y="0" width="{width}" height="{height}" rx="18" fill="#f8fafc"/>
      <line x1="{padding}" y1="{height-padding}" x2="{width-padding}" y2="{height-padding}" stroke="#cbd5e1" stroke-width="2"/>
      <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height-padding}" stroke="#cbd5e1" stroke-width="2"/>
      <polyline fill="none" stroke="{stroke_a}" stroke-width="4" points="{project(ys_train)}"/>
      <polyline fill="none" stroke="{stroke_b}" stroke-width="4" points="{project(ys_val)}"/>
      <text x="{padding}" y="18" fill="#0f172a" font-size="14" font-family="Arial, sans-serif">{title}</text>
      <text x="{width-180}" y="18" fill="{stroke_a}" font-size="12" font-family="Arial, sans-serif">train F1</text>
      <text x="{width-90}" y="18" fill="{stroke_b}" font-size="12" font-family="Arial, sans-serif">val F1</text>
    </svg>
    """


def architecture_svg() -> str:
    return """
    <svg width="760" height="135" viewBox="0 0 760 135" role="img" aria-label="Custom CNN architecture">
      <rect width="760" height="135" rx="18" fill="#f8fafc"/>
      <rect x="18" y="38" width="90" height="50" rx="12" fill="#dbeafe"/>
      <rect x="132" y="38" width="100" height="50" rx="12" fill="#bfdbfe"/>
      <rect x="256" y="38" width="100" height="50" rx="12" fill="#93c5fd"/>
      <rect x="380" y="38" width="100" height="50" rx="12" fill="#60a5fa"/>
      <rect x="504" y="38" width="86" height="50" rx="12" fill="#38bdf8"/>
      <rect x="614" y="38" width="126" height="50" rx="12" fill="#22c55e"/>
      <text x="63" y="67" text-anchor="middle" font-size="12">Mel input</text>
      <text x="182" y="67" text-anchor="middle" font-size="12">64 + SE</text>
      <text x="306" y="67" text-anchor="middle" font-size="12">128 + SE</text>
      <text x="430" y="67" text-anchor="middle" font-size="12">256 + SE</text>
      <text x="547" y="67" text-anchor="middle" font-size="12">GAP</text>
      <text x="677" y="67" text-anchor="middle" font-size="12">FC head / 10 classes</text>
      <line x1="108" y1="63" x2="132" y2="63" stroke="#334155" stroke-width="2.5"/>
      <line x1="232" y1="63" x2="256" y2="63" stroke="#334155" stroke-width="2.5"/>
      <line x1="356" y1="63" x2="380" y2="63" stroke="#334155" stroke-width="2.5"/>
      <line x1="480" y1="63" x2="504" y2="63" stroke="#334155" stroke-width="2.5"/>
      <line x1="590" y1="63" x2="614" y2="63" stroke="#334155" stroke-width="2.5"/>
    </svg>
    """


def pipeline_svg() -> str:
    return """
    <svg width="760" height="125" viewBox="0 0 760 125" role="img" aria-label="Project pipeline">
      <rect width="760" height="125" rx="18" fill="#fff7ed"/>
      <rect x="16" y="34" width="118" height="46" rx="12" fill="#fed7aa"/>
      <rect x="158" y="34" width="132" height="46" rx="12" fill="#fdba74"/>
      <rect x="314" y="34" width="138" height="46" rx="12" fill="#fb923c"/>
      <rect x="476" y="34" width="112" height="46" rx="12" fill="#f97316"/>
      <rect x="612" y="34" width="132" height="46" rx="12" fill="#ea580c"/>
      <text x="75" y="62" text-anchor="middle" font-size="12">Stem folders</text>
      <text x="224" y="62" text-anchor="middle" font-size="12">Mashup + ESC-50 noise</text>
      <text x="383" y="62" text-anchor="middle" font-size="12">Mel / AST features</text>
      <text x="532" y="62" text-anchor="middle" font-size="12">Three models</text>
      <text x="678" y="62" text-anchor="middle" font-size="12">TTA ensemble</text>
      <line x1="134" y1="57" x2="158" y2="57" stroke="#9a3412" stroke-width="2.5"/>
      <line x1="290" y1="57" x2="314" y2="57" stroke="#9a3412" stroke-width="2.5"/>
      <line x1="452" y1="57" x2="476" y2="57" stroke="#9a3412" stroke-width="2.5"/>
      <line x1="588" y1="57" x2="612" y2="57" stroke="#9a3412" stroke-width="2.5"/>
    </svg>
    """


def build_report_html(histories: dict[str, list[tuple[int, float, float]]], distribution: Counter) -> str:
    effnet_best = max(v for _, _, v in histories["effnet"])
    ast_best = max(v for _, _, v in histories["ast"])
    html_distribution = "".join(
        f"<tr><td>{genre}</td><td>{distribution.get(genre, 0)}</td></tr>" for genre in GENRES
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Messy Mashup Report</title>
  <style>
    @page {{
      size: A4;
      margin: 10mm;
    }}
    :root {{
      --ink: #0f172a;
      --muted: #475569;
      --paper: #f8fafc;
      --card: #ffffff;
      --blue: #2563eb;
      --teal: #0f766e;
      --orange: #c2410c;
      --border: #cbd5e1;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: "Segoe UI", Arial, sans-serif;
      background: linear-gradient(180deg, #e0f2fe 0%, #f8fafc 35%, #fff7ed 100%);
      color: var(--ink);
      margin: 0;
      padding: 12px;
    }}
    .page {{
      max-width: 1100px;
      margin: 0 auto;
      background: rgba(255,255,255,0.65);
    }}
    .hero {{
      background: linear-gradient(135deg, #0f172a, #1d4ed8 55%, #0f766e);
      color: white;
      border-radius: 22px;
      padding: 24px;
      margin-bottom: 14px;
    }}
    .hero h1 {{ margin: 0 0 8px; font-size: 30px; }}
    .hero p {{ margin: 6px 0; line-height: 1.35; max-width: 860px; font-size: 13px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 12px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid rgba(148, 163, 184, 0.35);
      border-radius: 18px;
      padding: 14px;
      box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
    }}
    .span-4 {{ grid-column: span 4; }}
    .span-5 {{ grid-column: span 5; }}
    .span-6 {{ grid-column: span 6; }}
    .span-7 {{ grid-column: span 7; }}
    .span-12 {{ grid-column: span 12; }}
    h2 {{ margin: 0 0 8px; font-size: 20px; }}
    h3 {{ margin: 0 0 8px; font-size: 16px; }}
    p, li {{ line-height: 1.4; color: var(--muted); font-size: 13px; }}
    ul {{ margin: 8px 0 0 18px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
      font-size: 12px;
    }}
    th, td {{
      border-bottom: 1px solid var(--border);
      padding: 5px 4px;
      text-align: left;
    }}
    .metric {{
      font-size: 24px;
      font-weight: 700;
      color: var(--ink);
    }}
    .metric-label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }}
    .callout {{
      background: #eff6ff;
      border-left: 5px solid var(--blue);
      padding: 10px 12px;
      border-radius: 12px;
      color: var(--ink);
    }}
    .warning {{
      background: #fff7ed;
      border-left: 5px solid var(--orange);
    }}
    .mono {{
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 13px;
    }}
    .small {{
      font-size: 12px;
    }}
    @media print {{
      body {{ padding: 0; background: white; }}
      .page {{ max-width: none; }}
      .card {{ box-shadow: none; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Messy Mashup</h1>
      <p><strong>Student:</strong> Sachin Kumar Ray (23F2002518)</p>
      <p>
        This project tackles noisy music genre classification for the IITM Deep Learning &amp; Generative AI project.
        The final system combines synthetic mashup generation, ESC-50 noise injection, spectrogram-based deep models,
        and test-time augmentation for robust leaderboard inference.
      </p>
      <p>
        The offline Kaggle solution is ensemble-based. The live deployment bundle serves a single EfficientNet classifier
        to keep CPU latency practical on Hugging Face Spaces.
      </p>
    </section>

    <section class="grid">
      <div class="card span-4">
        <div class="metric-label">Best archived EfficientNet val F1</div>
        <div class="metric">{effnet_best:.4f}</div>
        <p class="small">Recovered from the executed Kaggle notebook archive in the local workspace.</p>
      </div>
      <div class="card span-4">
        <div class="metric-label">Best archived AST val F1</div>
        <div class="metric">{ast_best:.4f}</div>
        <p class="small">Recovered from the same archived Kaggle execution snapshot.</p>
      </div>
      <div class="card span-4">
        <div class="metric-label">Final Kaggle score</div>
        <div class="metric">0.85477</div>
        <p class="small">Reference score from notebook version V26 supplied by the student.</p>
      </div>

      <div class="card span-7">
        <h2>Problem and Data</h2>
        <p>
          The task is to predict one of 10 music genres for each test clip in the Messy Mashup challenge. Training clips
          are provided as genre-specific stem folders, while the test split contains mixed and noisy audio where simple
          source-separation assumptions break down.
        </p>
        <div class="callout">
          Primary metric: macro F1 score. The pipeline uses mashup synthesis, ESC-50 noise injection, and TTA to reduce train-test mismatch.
        </div>
      </div>

      <div class="card span-5">
        <h2>Model Set</h2>
        <table>
          <thead>
            <tr><th>Model</th><th>Type</th><th>Status</th></tr>
          </thead>
          <tbody>
            <tr><td>EfficientNet-B0</td><td>Pretrained CNN</td><td>Archived metrics + checkpoints available</td></tr>
            <tr><td>Custom CNN + SE</td><td>From scratch</td><td>Code preserved, archived metrics unavailable</td></tr>
            <tr><td>AST</td><td>Pretrained transformer</td><td>Archived metrics + checkpoints available</td></tr>
          </tbody>
        </table>
        <p class="small">
          The scratch-model code remains in the canonical notebook to satisfy the course requirement, but the current local snapshot only
          preserved executed results for EfficientNet and AST.
        </p>
      </div>

      <div class="card span-12">
        <h2>Pipeline</h2>
        {pipeline_svg()}
      </div>

      <div class="card span-6">
        <h2>EfficientNet Training Curve</h2>
        {sparkline_svg(histories["effnet"], "EfficientNet F1 progression", "#2563eb", "#0f766e")}
      </div>

      <div class="card span-6">
        <h2>AST Training Curve</h2>
        {sparkline_svg(histories["ast"], "AST F1 progression", "#7c3aed", "#ea580c")}
      </div>

      <div class="card span-12">
        <h2>Custom CNN Architecture</h2>
        {architecture_svg()}
        <p>
          The scratch baseline uses stacked convolutional blocks with squeeze-and-excitation attention, global pooling, and a compact classification head.
        </p>
      </div>

      <div class="card span-6">
        <h2>Comparative Analysis</h2>
        <table>
          <thead>
            <tr><th>Model</th><th>Best val F1</th><th>Remarks</th></tr>
          </thead>
          <tbody>
            <tr><td>EfficientNet-B0</td><td>{effnet_best:.4f}</td><td>Strongest preserved checkpoint, best live deployment candidate</td></tr>
            <tr><td>Custom CNN + SE</td><td>Pending archival rerun</td><td>Notebook code present, no executed local metrics captured</td></tr>
            <tr><td>AST</td><td>{ast_best:.4f}</td><td>Strong transformer baseline, useful diversity in ensemble</td></tr>
          </tbody>
        </table>
        <p>
          The final Kaggle inference path uses weighted averaging with higher weight on EfficientNet because it showed the best preserved validation performance.
        </p>
        <p class="small">
          Final leaderboard reference used in this report: notebook <strong>V26</strong>, score <strong>0.85477</strong>.
        </p>
      </div>

      <div class="card span-6">
        <h2>Submission Distribution</h2>
        <table>
          <thead>
            <tr><th>Genre</th><th>Count</th></tr>
          </thead>
          <tbody>
            {html_distribution}
          </tbody>
        </table>
        <p class="small">
          Distribution extracted from the bundled `submission.csv` inside the archived Kaggle outputs.
        </p>
      </div>

      <div class="card span-7">
        <h2>Deployment</h2>
        <p>
          The deployment bundle uses a single EfficientNet checkpoint and the same mel-spectrogram preprocessing as the notebook.
          The app returns the predicted label, per-class probabilities, and a compact confidence table suitable for demo evaluation.
        </p>
        <ul>
          <li>Runtime target: Hugging Face Spaces with Gradio</li>
          <li>Model served live: EfficientNet-B0 checkpoint `effnet_f1.pt`</li>
          <li>Reason: lower CPU cost than the full ensemble while preserving strong validation quality</li>
        </ul>
      </div>

      <div class="card span-5">
        <h2>Error Analysis and Risks</h2>
        <div class="callout warning">
          The preserved archive does not include executed scratch-model metrics. The report therefore distinguishes between
          archived evidence and notebook-defined experimental intent instead of inventing missing numbers.
        </div>
        <ul>
          <li>Genre confusions are most likely between acoustically similar classes such as `rock`, `metal`, and `pop`.</li>
          <li>Heavy overlap, percussion bleed, and environmental noise remain the main failure sources.</li>
          <li>Future work: rerun the custom CNN, add waveform models, and calibrate the ensemble weights with full W&amp;B tracking.</li>
        </ul>
      </div>

      <div class="card span-12">
        <h2>References and Assets</h2>
        <p class="mono">
          Kaggle notebook: https://www.kaggle.com/code/godusssop/dl-23f2002518-notebook-t12026<br>
          Final Kaggle score reference: 0.85477 (V26)<br>
          GitHub repo: https://github.com/23f2002518/ml-web-project/<br>
          W&amp;B entity/project target: 23f2002518-dl-genai-project / 23f2002518-t12026<br>
          Hugging Face Space target: sachin-ray/messy-mashup-23f2002518
        </p>
      </div>
    </section>
  </div>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-notebook", type=Path, required=True)
    parser.add_argument("--results-zip", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("report/generated"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    notebook = load_notebook(args.archive_notebook)
    histories = extract_histories(notebook)
    distribution = extract_submission_distribution(args.results_zip)
    html = build_report_html(histories, distribution)

    output_html = args.output_dir / "report.html"
    output_html.write_text(html)
    print(output_html)
    print(args.output_dir / "23f2002518_DG_T12026.pdf")


if __name__ == "__main__":
    main()
