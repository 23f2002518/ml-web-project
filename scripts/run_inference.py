from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from project_pipeline import (
    MODEL_CHOICES,
    ExperimentConfig,
    build_label_encoder,
    build_loader,
    build_test_dataset,
    create_feature_extractor,
    create_model,
    get_device,
    load_test_metadata,
    model_hparams,
    predict_proba,
    seed_everything,
)


def load_metrics(metrics_path: Path) -> dict | None:
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text())


def load_trained_model(
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
    parser = argparse.ArgumentParser(description="Run ensemble inference for Messy Mashup.")
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
        help="Directory containing trained checkpoints and metrics.",
    )
    parser.add_argument(
        "--submission-out",
        type=Path,
        default=Path("artifacts/inference/submission.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = ExperimentConfig(base=args.base, output_dir=args.checkpoint_dir, seed=args.seed)
    seed_everything(config.seed)
    device = get_device(args.device)
    label_encoder = build_label_encoder(config)
    test_df = load_test_metadata(config)

    all_predictions = []
    all_weights = []
    file_ids = None

    for model_name in MODEL_CHOICES:
        feature_extractor = create_feature_extractor(model_name)
        hparams = model_hparams(model_name, config)
        metrics = load_metrics(args.checkpoint_dir / f"{model_name}_metrics.json")
        if not metrics:
            continue
        for fold in metrics.get("fold_metrics", []):
            checkpoint = Path(fold["checkpoint"])
            if not checkpoint.exists():
                continue
            model = load_trained_model(model_name, checkpoint, config, device)
            for shift in config.tta_shifts:
                test_dataset = build_test_dataset(
                    model_name=model_name,
                    test_df=test_df,
                    config=config,
                    feature_extractor=feature_extractor,
                    offset=shift,
                )
                loader = build_loader(test_dataset, hparams["batch_size"], shuffle=False, device=device)
                probs, ids = predict_proba(model, loader, device)
                all_predictions.append(probs)
                all_weights.append(float(fold["best_val_f1"]) * hparams["weight_scale"])
                if file_ids is None:
                    file_ids = ids

    if not all_predictions or file_ids is None:
        raise SystemExit(
            "No checkpoints were found. Run scripts/train_models.py first or point --checkpoint-dir to existing artifacts."
        )

    weights = np.array(all_weights, dtype=np.float32)
    weights = weights / weights.sum()

    ensemble = np.zeros_like(all_predictions[0])
    for weight, prediction in zip(weights, all_predictions):
        ensemble += weight * prediction

    final_preds = np.argmax(ensemble, axis=1)
    final_genres = label_encoder.inverse_transform(final_preds)
    submission = pd.DataFrame({"id": file_ids, "genre": final_genres}).sort_values("id").reset_index(drop=True)
    args.submission_out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.submission_out, index=False)

    summary = {
        "models_used": len(all_predictions),
        "unique_ids": len(submission),
        "submission_path": str(args.submission_out),
    }
    (args.submission_out.parent / "inference_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved submission to {args.submission_out}")


if __name__ == "__main__":
    main()
