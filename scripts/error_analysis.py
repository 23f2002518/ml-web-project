from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay

from project_pipeline import (
    MODEL_CHOICES,
    ExperimentConfig,
    build_label_encoder,
    build_loader,
    build_train_val_datasets,
    classification_payload,
    collect_eval_predictions,
    create_feature_extractor,
    create_model,
    create_splitter,
    discover_noise_files,
    get_device,
    load_training_metadata,
    model_hparams,
    seed_everything,
    write_json,
)


def load_checkpoint(
    model_name: str,
    checkpoint_path: Path,
    config: ExperimentConfig,
    device: torch.device,
) -> torch.nn.Module:
    model = create_model(model_name, config).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate validation-side error analysis artifacts.")
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup"),
        help="Path to the Messy Mashup dataset root.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("artifacts/training"),
        help="Directory containing trained checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/error-analysis"),
        help="Directory for analysis outputs.",
    )
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = ExperimentConfig(base=args.base, output_dir=args.checkpoint_dir, seed=args.seed)
    seed_everything(config.seed)
    device = get_device(args.device)
    label_encoder = build_label_encoder(config)
    train_df, genre_songs = load_training_metadata(config, label_encoder)
    noise_files = discover_noise_files(config)
    feature_extractor = create_feature_extractor(args.model)
    splitter = create_splitter(config)
    hparams = model_hparams(args.model, config)

    rows = []
    all_labels = []
    all_preds = []

    for fold, (_, val_idx) in enumerate(splitter.split(train_df, train_df["label"])):
        if fold >= hparams["folds"]:
            break
        checkpoint_path = args.checkpoint_dir / f"{hparams['checkpoint_prefix']}_f{fold}.pt"
        if not checkpoint_path.exists():
            continue

        val_subset = train_df.iloc[val_idx].copy()
        _, val_dataset = build_train_val_datasets(
            model_name=args.model,
            train_df=train_df.iloc[val_idx],
            val_df=val_subset,
            genre_songs=genre_songs,
            noise_files=noise_files,
            config=config,
            feature_extractor=feature_extractor,
        )
        val_loader = build_loader(val_dataset, hparams["batch_size"], shuffle=False, device=device)
        model = load_checkpoint(args.model, checkpoint_path, config, device)
        labels, preds, probs = collect_eval_predictions(model, val_loader, device)

        all_labels.extend(labels.tolist())
        all_preds.extend(preds.tolist())
        predicted_genres = label_encoder.inverse_transform(preds)
        actual_genres = label_encoder.inverse_transform(labels)
        confidences = probs.max(axis=1)
        fold_rows = val_subset.reset_index(drop=True).copy()
        fold_rows["fold"] = fold
        fold_rows["actual_genre"] = actual_genres
        fold_rows["predicted_genre"] = predicted_genres
        fold_rows["confidence"] = confidences
        fold_rows["correct"] = fold_rows["actual_genre"] == fold_rows["predicted_genre"]
        rows.append(fold_rows)

    if not rows:
        raise SystemExit(
            "No checkpoints were found for error analysis. Run scripts/train_models.py first."
        )

    analysis_dir = args.output_dir / args.model
    analysis_dir.mkdir(parents=True, exist_ok=True)

    payload = classification_payload(
        labels=pd.Series(all_labels).to_numpy(),
        preds=pd.Series(all_preds).to_numpy(),
        label_encoder=label_encoder,
    )
    write_json(analysis_dir / "classification_report.json", payload)

    matrix = payload["confusion_matrix"]
    matrix_df = pd.DataFrame(matrix, index=label_encoder.classes_, columns=label_encoder.classes_)
    matrix_df.to_csv(analysis_dir / "confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(matrix_df.to_numpy(), display_labels=label_encoder.classes_).plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        colorbar=False,
    )
    fig.tight_layout()
    fig.savefig(analysis_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    all_rows = pd.concat(rows, ignore_index=True)
    misclassified = all_rows.loc[~all_rows["correct"]].sort_values(["fold", "actual_genre", "predicted_genre"])
    misclassified.to_csv(analysis_dir / "misclassified_examples.csv", index=False)

    summary = {
        "analysis_dir": str(analysis_dir),
        "total_examples": int(len(all_rows)),
        "misclassified_examples": int(len(misclassified)),
        "macro_f1": payload["macro_f1"],
        "accuracy": payload["accuracy"],
    }
    write_json(analysis_dir / "summary.json", summary)
    print(f"Saved error analysis outputs to {analysis_dir}")


if __name__ == "__main__":
    main()
