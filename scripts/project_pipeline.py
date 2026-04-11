from __future__ import annotations

import gc
import os
import random
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import ASTFeatureExtractor, ASTForAudioClassification


warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

WANDB_ENTITY_DEFAULT = "23f2002518-dl-genai-project"
WANDB_PROJECT_DEFAULT = "23f2002518-t12026"
MODEL_CHOICES = ("efficientnet-b0", "custom-cnn", "ast-transformer")
GENRES = (
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
)


@dataclass
class ExperimentConfig:
    base: Path
    output_dir: Path = Path("artifacts/training")
    sr: int = 16000
    duration: int = 10
    n_mels: int = 128
    n_fft: int = 1024
    hop: int = 256
    seed: int = 42
    n_folds: int = 5
    train_folds: int = 5
    custom_cnn_folds: int = 1
    epochs_effnet: int = 30
    epochs_cnn: int = 20
    epochs_ast: int = 20
    batch_effnet: int = 16
    batch_cnn: int = 16
    batch_ast: int = 8
    lr_effnet: float = 5e-4
    lr_cnn: float = 1e-3
    lr_ast: float = 5e-6
    mashup_prob: float = 0.7
    noise_prob: float = 0.5
    snr_min: int = 8
    snr_max: int = 28
    tta_shifts: tuple[float, ...] = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)
    genres: tuple[str, ...] = field(default_factory=lambda: GENRES)
    num_workers: int = 0

    @property
    def n_samples(self) -> int:
        return self.sr * self.duration

    @property
    def num_classes(self) -> int:
        return len(self.genres)

    @property
    def train_path(self) -> Path:
        return self.base / "genres_stems"

    @property
    def noise_path(self) -> Path:
        return self.base / "ESC-50-master" / "audio"

    @property
    def test_csv(self) -> Path:
        return self.base / "test.csv"

    def to_run_config(self) -> dict[str, Any]:
        data = asdict(self)
        data["base"] = str(self.base)
        data["output_dir"] = str(self.output_dir)
        data["tta_shifts"] = list(self.tta_shifts)
        data["genres"] = list(self.genres)
        return data


def get_device(explicit: str | None = None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def build_label_encoder(config: ExperimentConfig) -> LabelEncoder:
    return LabelEncoder().fit(list(config.genres))


def load_training_metadata(
    config: ExperimentConfig,
    label_encoder: LabelEncoder,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    train_rows: list[dict[str, str]] = []
    genre_songs: dict[str, list[str]] = defaultdict(list)
    for genre in config.genres:
        genre_dir = config.train_path / genre
        if not genre_dir.exists():
            continue
        for song in sorted(genre_dir.iterdir()):
            if song.is_dir():
                row = {"path": str(song), "genre": genre}
                train_rows.append(row)
                genre_songs[genre].append(str(song))
    train_df = pd.DataFrame(train_rows)
    if train_df.empty:
        raise FileNotFoundError(
            f"No training stems found under {config.train_path}. "
            "Point --base to the Messy Mashup dataset root."
        )
    train_df["label"] = label_encoder.transform(train_df["genre"])
    return train_df, genre_songs


def load_test_metadata(config: ExperimentConfig) -> pd.DataFrame:
    if not config.test_csv.exists():
        raise FileNotFoundError(f"Missing test.csv at {config.test_csv}")
    return pd.read_csv(config.test_csv)


def discover_noise_files(config: ExperimentConfig) -> list[Path]:
    if not config.noise_path.exists():
        return []
    return sorted(config.noise_path.glob("*.wav"))


def load_audio(
    path: Path | str,
    config: ExperimentConfig,
    offset: float = 0.0,
) -> np.ndarray:
    target = config.n_samples
    try:
        audio, _ = librosa.load(
            str(path),
            sr=config.sr,
            duration=config.duration,
            offset=max(0.0, offset),
        )
    except Exception:
        return np.zeros(target, dtype=np.float32)
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    else:
        audio = audio[:target]
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak
    return audio.astype(np.float32)


def load_stems(song_path: str | Path, config: ExperimentConfig) -> dict[str, np.ndarray]:
    stems: dict[str, np.ndarray] = {}
    song_dir = Path(song_path)
    for stem_name in ("drums", "vocals", "bass", "others"):
        stem_file = song_dir / f"{stem_name}.wav"
        if stem_file.exists():
            stems[stem_name] = load_audio(stem_file, config)
        else:
            stems[stem_name] = np.zeros(config.n_samples, dtype=np.float32)
    return stems


def combine_stems(stems: dict[str, np.ndarray]) -> np.ndarray:
    combined = np.zeros_like(next(iter(stems.values())))
    for stem in stems.values():
        combined += stem
    peak = np.max(np.abs(combined))
    if peak > 1e-6:
        combined = combined / peak
    return combined.astype(np.float32)


def create_mashup(
    genre: str,
    genre_songs: dict[str, list[str]],
    config: ExperimentConfig,
) -> np.ndarray:
    songs = genre_songs[genre]
    if len(songs) < 4:
        stems = load_stems(random.choice(songs), config)
        weights = np.random.uniform(0.5, 1.5, len(stems))
        mashup = sum(stem * weight for stem, weight in zip(stems.values(), weights))
    else:
        selected = random.sample(songs, 4)
        mashup = np.zeros(config.n_samples, dtype=np.float32)
        for idx, stem_name in enumerate(("drums", "vocals", "bass", "others")):
            stem_path = Path(selected[idx]) / f"{stem_name}.wav"
            if stem_path.exists():
                mashup += np.random.uniform(0.5, 1.5) * load_audio(stem_path, config)
    peak = np.max(np.abs(mashup))
    if peak > 1e-6:
        mashup = mashup / peak
    return mashup.astype(np.float32)


def add_noise(
    audio: np.ndarray,
    noise_files: list[Path],
    config: ExperimentConfig,
) -> np.ndarray:
    if not noise_files or random.random() > config.noise_prob:
        return audio
    snr = np.random.uniform(config.snr_min, config.snr_max)
    noise_file = random.choice(noise_files)
    try:
        noise, _ = librosa.load(str(noise_file), sr=config.sr)
    except Exception:
        return audio
    while len(noise) < len(audio):
        noise = np.tile(noise, 2)
    start = random.randint(0, max(0, len(noise) - len(audio)))
    noise = noise[start : start + len(audio)]
    if len(noise) < len(audio):
        noise = np.pad(noise, (0, len(audio) - len(noise)))
    audio_power = np.mean(audio**2) + 1e-10
    noise_power = np.mean(noise**2) + 1e-10
    scale = np.sqrt(audio_power / (10 ** (snr / 10) * noise_power))
    mixed = audio + scale * noise
    peak = np.max(np.abs(mixed))
    if peak > 1e-6:
        mixed = mixed / peak
    return mixed.astype(np.float32)


def to_mel(audio: np.ndarray, config: ExperimentConfig) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=config.sr,
        n_mels=config.n_mels,
        n_fft=config.n_fft,
        hop_length=config.hop,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return ((mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)).astype(np.float32)


class MelTrainDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        genre_songs: dict[str, list[str]],
        noise_files: list[Path],
        config: ExperimentConfig,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.genre_songs = genre_songs
        self.noise_files = noise_files
        self.config = config

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        if random.random() < self.config.mashup_prob:
            audio = create_mashup(row["genre"], self.genre_songs, self.config)
        else:
            stems = load_stems(row["path"], self.config)
            weights = np.random.uniform(0.5, 1.5, len(stems))
            audio = sum(stem * weight for stem, weight in zip(stems.values(), weights))
            peak = np.max(np.abs(audio))
            if peak > 1e-6:
                audio = audio / peak
        audio = add_noise(audio.astype(np.float32), self.noise_files, self.config)
        features = torch.from_numpy(to_mel(audio, self.config)).unsqueeze(0)
        label = torch.tensor(row["label"], dtype=torch.long)
        return features, label


class MelEvalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, config: ExperimentConfig) -> None:
        self.df = df.reset_index(drop=True)
        self.config = config

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        stems = load_stems(row["path"], self.config)
        audio = combine_stems(stems)
        features = torch.from_numpy(to_mel(audio, self.config)).unsqueeze(0)
        label = torch.tensor(row["label"], dtype=torch.long)
        return features, label


class TestMelDataset(Dataset):
    def __init__(self, df: pd.DataFrame, config: ExperimentConfig, offset: float = 0.0) -> None:
        self.df = df.reset_index(drop=True)
        self.config = config
        self.offset = offset

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        audio = load_audio(self.config.base / row["filename"], self.config, offset=self.offset)
        features = torch.from_numpy(to_mel(audio, self.config)).unsqueeze(0)
        return features, int(row["id"])


class ASTTrainDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        genre_songs: dict[str, list[str]],
        noise_files: list[Path],
        feature_extractor: ASTFeatureExtractor,
        config: ExperimentConfig,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.genre_songs = genre_songs
        self.noise_files = noise_files
        self.feature_extractor = feature_extractor
        self.config = config

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        if random.random() < self.config.mashup_prob:
            audio = create_mashup(row["genre"], self.genre_songs, self.config)
        else:
            stems = load_stems(row["path"], self.config)
            weights = np.random.uniform(0.5, 1.5, len(stems))
            audio = sum(stem * weight for stem, weight in zip(stems.values(), weights))
            peak = np.max(np.abs(audio))
            if peak > 1e-6:
                audio = audio / peak
        audio = add_noise(audio.astype(np.float32), self.noise_files, self.config)
        inputs = self.feature_extractor(
            audio,
            sampling_rate=self.config.sr,
            return_tensors="pt",
            padding="max_length",
            max_length=1024,
        )
        label = torch.tensor(row["label"], dtype=torch.long)
        return inputs.input_values.squeeze(0), label


class ASTEvalDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_extractor: ASTFeatureExtractor,
        config: ExperimentConfig,
        offset: float = 0.0,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.feature_extractor = feature_extractor
        self.config = config
        self.offset = offset

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        stems = load_stems(row["path"], self.config)
        audio = combine_stems(stems)
        inputs = self.feature_extractor(
            audio,
            sampling_rate=self.config.sr,
            return_tensors="pt",
            padding="max_length",
            max_length=1024,
        )
        label = torch.tensor(row["label"], dtype=torch.long)
        return inputs.input_values.squeeze(0), label


class ASTTestDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_extractor: ASTFeatureExtractor,
        config: ExperimentConfig,
        offset: float = 0.0,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.feature_extractor = feature_extractor
        self.config = config
        self.offset = offset

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        audio = load_audio(self.config.base / row["filename"], self.config, offset=self.offset)
        inputs = self.feature_extractor(
            audio,
            sampling_rate=self.config.sr,
            return_tensors="pt",
            padding="max_length",
            max_length=1024,
        )
        return inputs.input_values.squeeze(0), int(row["id"])


class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            in_chans=1,
            num_classes=0,
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            SEBlock(out_channels),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.pool(self.features(x)))


class ASTWrapper(nn.Module):
    def __init__(self, model: ASTForAudioClassification) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).logits


def get_ast_name() -> str:
    return "MIT/ast-finetuned-audioset-10-10-0.4593"


def create_model(model_name: str, config: ExperimentConfig) -> nn.Module:
    if model_name == "efficientnet-b0":
        return EfficientNetClassifier(num_classes=config.num_classes)
    if model_name == "custom-cnn":
        return CustomCNN(num_classes=config.num_classes)
    if model_name == "ast-transformer":
        ast_name = get_ast_name()
        ast_base = ASTForAudioClassification.from_pretrained(
            ast_name,
            num_labels=config.num_classes,
            ignore_mismatched_sizes=True,
        )
        for parameter in ast_base.audio_spectrogram_transformer.embeddings.parameters():
            parameter.requires_grad = False
        return ASTWrapper(ast_base)
    raise ValueError(f"Unsupported model: {model_name}")


def create_feature_extractor(model_name: str) -> ASTFeatureExtractor | None:
    if model_name != "ast-transformer":
        return None
    return ASTFeatureExtractor.from_pretrained(get_ast_name())


def model_hparams(model_name: str, config: ExperimentConfig) -> dict[str, Any]:
    if model_name == "efficientnet-b0":
        return {
            "epochs": config.epochs_effnet,
            "batch_size": config.batch_effnet,
            "lr": config.lr_effnet,
            "folds": config.train_folds,
            "checkpoint_prefix": "effnet",
            "weight_scale": 1.5,
        }
    if model_name == "custom-cnn":
        return {
            "epochs": config.epochs_cnn,
            "batch_size": config.batch_cnn,
            "lr": config.lr_cnn,
            "folds": config.custom_cnn_folds,
            "checkpoint_prefix": "cnn",
            "weight_scale": 0.5,
        }
    if model_name == "ast-transformer":
        return {
            "epochs": config.epochs_ast,
            "batch_size": config.batch_ast,
            "lr": config.lr_ast,
            "folds": config.train_folds,
            "checkpoint_prefix": "ast",
            "weight_scale": 1.0,
        }
    raise ValueError(f"Unsupported model: {model_name}")


def init_wandb_run(
    model_name: str,
    config: ExperimentConfig,
    enabled: bool = True,
    extra_config: dict[str, Any] | None = None,
    notes: str | None = None,
):
    if not enabled or not os.environ.get("WANDB_API_KEY"):
        return None
    wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
    run_config = config.to_run_config()
    run_config["model_name"] = model_name
    if extra_config:
        run_config.update(extra_config)
    return wandb.init(
        entity=os.environ.get("WANDB_ENTITY", WANDB_ENTITY_DEFAULT),
        project=os.environ.get("WANDB_PROJECT", WANDB_PROJECT_DEFAULT),
        name=model_name,
        config=run_config,
        notes=notes,
        reinit=True,
    )


def init_comparison_run(enabled: bool = True, notes: str | None = None):
    if not enabled or not os.environ.get("WANDB_API_KEY"):
        return None
    wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
    return wandb.init(
        entity=os.environ.get("WANDB_ENTITY", WANDB_ENTITY_DEFAULT),
        project=os.environ.get("WANDB_PROJECT", WANDB_PROJECT_DEFAULT),
        name="model-comparison",
        notes=notes,
        reinit=True,
    )


def finish_wandb_run(summary: dict[str, Any] | None = None) -> None:
    if summary:
        for key, value in summary.items():
            wandb.summary[key] = value
    if wandb.run is not None:
        wandb.finish()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    device: torch.device,
) -> tuple[float, float, float]:
    model.train()
    losses: list[float] = []
    preds: list[int] = []
    labels: list[int] = []
    for features, target in tqdm(loader, leave=False):
        features = features.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        preds.extend(logits.argmax(1).cpu().numpy())
        labels.extend(target.cpu().numpy())
    return np.mean(losses), f1_score(labels, preds, average="macro"), accuracy_score(labels, preds)


@torch.no_grad()
def valid_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    losses: list[float] = []
    preds: list[int] = []
    labels: list[int] = []
    for features, target in tqdm(loader, leave=False):
        features = features.to(device)
        target = target.to(device)
        logits = model(features)
        loss = criterion(logits, target)
        losses.append(loss.item())
        preds.extend(logits.argmax(1).cpu().numpy())
        labels.extend(target.cpu().numpy())
    return np.mean(losses), f1_score(labels, preds, average="macro"), accuracy_score(labels, preds)


@torch.no_grad()
def predict_proba(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, list[int]]:
    model.eval()
    all_probs: list[np.ndarray] = []
    all_ids: list[int] = []
    for features, ids in tqdm(loader, leave=False):
        features = features.to(device)
        probs = F.softmax(model(features), dim=1).cpu().numpy()
        all_probs.extend(probs)
        for item in ids:
            all_ids.append(item.item() if isinstance(item, torch.Tensor) else int(item))
    return np.array(all_probs), all_ids


@torch.no_grad()
def collect_eval_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels: list[int] = []
    preds: list[int] = []
    probs: list[np.ndarray] = []
    for features, target in tqdm(loader, leave=False):
        features = features.to(device)
        target = target.to(device)
        logits = model(features)
        fold_probs = F.softmax(logits, dim=1).cpu().numpy()
        probs.extend(fold_probs)
        preds.extend(np.argmax(fold_probs, axis=1))
        labels.extend(target.cpu().numpy())
    return np.array(labels), np.array(preds), np.array(probs)


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )


def build_train_val_datasets(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    genre_songs: dict[str, list[str]],
    noise_files: list[Path],
    config: ExperimentConfig,
    feature_extractor: ASTFeatureExtractor | None = None,
) -> tuple[Dataset, Dataset]:
    if model_name == "ast-transformer":
        assert feature_extractor is not None
        return (
            ASTTrainDataset(train_df, genre_songs, noise_files, feature_extractor, config),
            ASTEvalDataset(val_df, feature_extractor, config),
        )
    return (
        MelTrainDataset(train_df, genre_songs, noise_files, config),
        MelEvalDataset(val_df, config),
    )


def build_test_dataset(
    model_name: str,
    test_df: pd.DataFrame,
    config: ExperimentConfig,
    feature_extractor: ASTFeatureExtractor | None = None,
    offset: float = 0.0,
) -> Dataset:
    if model_name == "ast-transformer":
        assert feature_extractor is not None
        return ASTTestDataset(test_df, feature_extractor, config, offset=offset)
    return TestMelDataset(test_df, config, offset=offset)


def create_splitter(config: ExperimentConfig) -> StratifiedKFold:
    return StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)


def write_json(path: Path, payload: Any) -> None:
    import json

    def _default(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_default))


def cleanup_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def classification_payload(
    labels: np.ndarray,
    preds: np.ndarray,
    label_encoder: LabelEncoder,
) -> dict[str, Any]:
    report = classification_report(
        labels,
        preds,
        target_names=list(label_encoder.classes_),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(labels, preds)
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "accuracy": accuracy_score(labels, preds),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }
