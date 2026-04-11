from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from project_pipeline import (
    MODEL_CHOICES,
    ExperimentConfig,
    build_label_encoder,
    build_loader,
    build_train_val_datasets,
    cleanup_memory,
    create_feature_extractor,
    create_model,
    create_splitter,
    discover_noise_files,
    finish_wandb_run,
    get_device,
    init_comparison_run,
    init_wandb_run,
    load_training_metadata,
    model_hparams,
    seed_everything,
    train_epoch,
    valid_epoch,
    write_json,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def train_model(
    model_name: str,
    config: ExperimentConfig,
    train_df,
    genre_songs,
    noise_files,
    device: torch.device,
    enable_wandb: bool,
) -> dict:
    hparams = model_hparams(model_name, config)
    label_encoder = build_label_encoder(config)
    feature_extractor = create_feature_extractor(model_name)
    splitter = create_splitter(config)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run = init_wandb_run(
        model_name=model_name,
        config=config,
        enabled=enable_wandb,
        extra_config={
            "learning_rate": hparams["lr"],
            "epochs": hparams["epochs"],
            "batch_size": hparams["batch_size"],
            "folds": hparams["folds"],
        },
        notes="Training run launched from the script-based pipeline.",
    )

    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(train_df, train_df["label"])):
        if fold >= hparams["folds"]:
            break

        train_subset = train_df.iloc[train_idx]
        val_subset = train_df.iloc[val_idx]
        train_ds, val_ds = build_train_val_datasets(
            model_name=model_name,
            train_df=train_subset,
            val_df=val_subset,
            genre_songs=genre_songs,
            noise_files=noise_files,
            config=config,
            feature_extractor=feature_extractor,
        )
        train_loader = build_loader(train_ds, hparams["batch_size"], shuffle=True, device=device)
        val_loader = build_loader(val_ds, hparams["batch_size"], shuffle=False, device=device)

        model = create_model(model_name, config).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        parameters = model.parameters()
        if model_name == "ast-transformer":
            parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = AdamW(parameters, lr=hparams["lr"], weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, hparams["epochs"])

        best_val_f1 = 0.0
        best_checkpoint = output_dir / f"{hparams['checkpoint_prefix']}_f{fold}.pt"
        patience = 10 if model_name == "custom-cnn" else 7
        if model_name == "ast-transformer":
            patience = 5
        no_improve = 0

        for epoch in range(hparams["epochs"]):
            train_loss, train_f1, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_f1, val_acc = valid_epoch(model, val_loader, criterion, device)
            scheduler.step()

            print(
                f"[{model_name}] fold {fold + 1} epoch {epoch + 1:02d} "
                f"TF1={train_f1:.4f} VF1={val_f1:.4f} TAcc={train_acc:.4f} VAcc={val_acc:.4f}"
            )

            if run is not None:
                run.log(
                    {
                        f"fold{fold}/train_loss": train_loss,
                        f"fold{fold}/train_f1": train_f1,
                        f"fold{fold}/train_acc": train_acc,
                        f"fold{fold}/val_loss": val_loss,
                        f"fold{fold}/val_f1": val_f1,
                        f"fold{fold}/val_acc": val_acc,
                    }
                )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), best_checkpoint)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"[{model_name}] early stop on fold {fold + 1} at epoch {epoch + 1}")
                    break

        fold_metrics.append(
            {
                "fold": fold,
                "best_val_f1": best_val_f1,
                "checkpoint": str(best_checkpoint),
                "val_size": int(len(val_subset)),
                "labels": {genre: int(count) for genre, count in val_subset["genre"].value_counts().sort_index().items()},
            }
        )
        cleanup_memory()

    summary = {
        "model": model_name,
        "num_classes": len(label_encoder.classes_),
        "folds_trained": len(fold_metrics),
        "mean_best_val_f1": float(np.mean([fold["best_val_f1"] for fold in fold_metrics])) if fold_metrics else None,
        "std_best_val_f1": float(np.std([fold["best_val_f1"] for fold in fold_metrics])) if len(fold_metrics) > 1 else 0.0,
        "fold_metrics": fold_metrics,
        "checkpoint_prefix": hparams["checkpoint_prefix"],
        "weight_scale": hparams["weight_scale"],
    }
    write_json(output_dir / f"{model_name}_metrics.json", summary)

    finish_summary = {}
    if summary["mean_best_val_f1"] is not None:
        finish_summary["best_val_f1"] = summary["mean_best_val_f1"]
        finish_summary["val_f1_std"] = summary["std_best_val_f1"]
    finish_wandb_run(finish_summary if run is not None else None)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Messy Mashup models outside the notebook.")
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup"),
        help="Path to the Messy Mashup dataset root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/training"),
        help="Directory for checkpoints and metrics.",
    )
    parser.add_argument(
        "--model",
        choices=("all",) + MODEL_CHOICES,
        default="all",
        help="Which model family to train.",
    )
    parser.add_argument("--device", default=None, help="Optional torch device override, e.g. cuda:0 or cpu.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-folds", type=int, default=5, help="Folds for EfficientNet and AST.")
    parser.add_argument("--custom-cnn-folds", type=int, default=1, help="Folds for the scratch model.")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable W&B logging even if credentials exist.")
    args = parser.parse_args()

    config = ExperimentConfig(
        base=args.base,
        output_dir=args.output_dir,
        seed=args.seed,
        train_folds=args.train_folds,
        custom_cnn_folds=args.custom_cnn_folds,
    )
    seed_everything(config.seed)
    device = get_device(args.device)

    label_encoder = build_label_encoder(config)
    train_df, genre_songs = load_training_metadata(config, label_encoder)
    noise_files = discover_noise_files(config)

    selected_models = MODEL_CHOICES if args.model == "all" else (args.model,)
    summaries = []
    for model_name in selected_models:
        summaries.append(
            train_model(
                model_name=model_name,
                config=config,
                train_df=train_df,
                genre_songs=genre_songs,
                noise_files=noise_files,
                device=device,
                enable_wandb=not args.disable_wandb,
            )
        )

    comparison = {
        "models": {summary["model"]: summary for summary in summaries},
        "notes": (
            "These metrics come from the script-based training pipeline. "
            "Use this output to create fully reproducible W&B evidence, including a real custom-CNN run."
        ),
    }
    write_json(config.output_dir / "comparison.json", comparison)

    run = init_comparison_run(
        enabled=not args.disable_wandb,
        notes="Comparison summary from script-based training runs.",
    )
    if run is not None:
        payload = {}
        for summary in summaries:
            if summary["mean_best_val_f1"] is not None:
                payload[f"comparison/{summary['model']}_f1"] = summary["mean_best_val_f1"]
        run.log(payload)
        finish_wandb_run()

    print("\nTraining summaries:")
    for summary in summaries:
        print(
            f"- {summary['model']}: folds={summary['folds_trained']} "
            f"mean_best_val_f1={summary['mean_best_val_f1']}"
        )


if __name__ == "__main__":
    main()
