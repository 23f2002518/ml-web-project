from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import wandb


EFFNET_PATTERN = re.compile(r"Ep\s*(\d+): TF1=(\d+\.\d+), VF1=(\d+\.\d+)")
AST_PATTERN = re.compile(r"Ep\s*(\d+): TF1=(\d+\.\d+), VF1=(\d+\.\d+)")


def load_notebook(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def cell_text(cell: dict) -> str:
    outputs = cell.get("outputs", [])
    parts: list[str] = []
    for output in outputs:
        if "text" in output:
            parts.extend(output["text"])
        elif "data" in output and "text/plain" in output["data"]:
            parts.extend(output["data"]["text/plain"])
    return "".join(parts)


def extract_archived_histories(notebook: dict) -> dict[str, list[dict[str, float]]]:
    histories = {"efficientnet-b0": [], "ast-transformer": []}
    for idx, cell in enumerate(notebook.get("cells", [])):
        text = cell_text(cell)
        if idx == 8 and "TRAINING EFFICIENTNET" in text:
            for match in EFFNET_PATTERN.finditer(text):
                histories["efficientnet-b0"].append(
                    {
                        "epoch": int(match.group(1)),
                        "train_f1": float(match.group(2)),
                        "val_f1": float(match.group(3)),
                    }
                )
        if idx == 9 and "TRAINING AST" in text:
            for match in AST_PATTERN.finditer(text):
                histories["ast-transformer"].append(
                    {
                        "epoch": int(match.group(1)),
                        "train_f1": float(match.group(2)),
                        "val_f1": float(match.group(3)),
                    }
                )
    return histories


def log_run(
    entity: str,
    project: str,
    name: str,
    config: dict,
    history: list[dict[str, float]],
    notes: str,
) -> None:
    run = wandb.init(
        entity=entity,
        project=project,
        name=name,
        config=config,
        notes=notes,
        reinit=True,
    )
    for step, row in enumerate(history):
        payload = dict(row)
        payload["step"] = step
        wandb.log(payload)
    if history:
        wandb.summary["best_val_f1"] = max(row["val_f1"] for row in history)
        wandb.summary["last_val_f1"] = history[-1]["val_f1"]
    else:
        wandb.summary["archived_metrics_available"] = False
    wandb.finish()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-notebook", type=Path, required=True)
    parser.add_argument("--results-zip", type=Path, required=False)
    parser.add_argument(
        "--entity",
        default=os.environ.get("WANDB_ENTITY", "23f2002518-dl-genai-project"),
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("WANDB_PROJECT", "23f2002518-t12026"),
    )
    args = parser.parse_args()

    if not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("WANDB_API_KEY is required.")

    notebook = load_notebook(args.archive_notebook)
    histories = extract_archived_histories(notebook)

    log_run(
        args.entity,
        args.project,
        "efficientnet-b0",
        {
            "model": "efficientnet_b0",
            "input": "mel_spectrogram",
            "source": "archived_kaggle_output",
        },
        histories["efficientnet-b0"],
        "Backfilled from the preserved Kaggle notebook output archive.",
    )

    log_run(
        args.entity,
        args.project,
        "custom-cnn",
        {
            "model": "custom_cnn_se",
            "input": "mel_spectrogram",
            "source": "repo_notebook_only",
            "archived_metrics_available": False,
        },
        [],
        (
            "Scratch-model code is preserved in the notebook, but the local snapshot did not "
            "include executed custom-CNN metrics. This run is a scaffold for later retraining."
        ),
    )

    log_run(
        args.entity,
        args.project,
        "ast-transformer",
        {
            "model": "MIT/ast-finetuned-audioset-10-10-0.4593",
            "input": "audio_spectrogram_transformer",
            "source": "archived_kaggle_output",
        },
        histories["ast-transformer"],
        "Backfilled from the preserved Kaggle notebook output archive.",
    )

    comparison = wandb.init(
        entity=args.entity,
        project=args.project,
        name="model-comparison",
        notes="Comparison summary generated from archived local experiment evidence.",
        reinit=True,
    )
    if histories["efficientnet-b0"]:
        wandb.summary["comparison/efficientnet_best_val_f1"] = max(
            row["val_f1"] for row in histories["efficientnet-b0"]
        )
    if histories["ast-transformer"]:
        wandb.summary["comparison/ast_best_val_f1"] = max(
            row["val_f1"] for row in histories["ast-transformer"]
        )
    wandb.summary["comparison/customcnn_best_val_f1"] = None
    wandb.summary["comparison/customcnn_status"] = "needs_rerun"
    wandb.finish()


if __name__ == "__main__":
    main()
