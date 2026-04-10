from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import timm
import torch
import torch.nn as nn
from gradio_client import utils as gradio_client_utils


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

SR = 16000
DURATION = 10
N_SAMPLES = SR * DURATION
N_MELS = 128
N_FFT = 1024
HOP = 256
CHECKPOINT_PATH = Path(__file__).resolve().parent / "artifacts" / "effnet_f1.pt"


def patch_gradio_schema_bug() -> None:
    """Handle Gradio's boolean-schema bug for file/audio components on Spaces."""
    if getattr(gradio_client_utils, "_messy_mashup_schema_patch", False):
        return

    original_get_type = gradio_client_utils.get_type

    def patched_get_type(schema):
        if isinstance(schema, bool):
            return "boolean"
        return original_get_type(schema)

    gradio_client_utils.get_type = patched_get_type
    gradio_client_utils._messy_mashup_schema_patch = True


patch_gradio_schema_bug()


class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
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


def load_audio(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SR, duration=DURATION)
    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)))
    else:
        audio = audio[:N_SAMPLES]
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak
    return audio.astype(np.float32)


def to_mel(audio: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_db.astype(np.float32)


def load_model() -> EfficientNetClassifier:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}.")
    model = EfficientNetClassifier(num_classes=len(GENRES))
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


MODEL: EfficientNetClassifier | None = None
MODEL_LOAD_ERROR: str | None = None

try:
    MODEL = load_model()
except Exception as exc:  # pragma: no cover
    MODEL_LOAD_ERROR = str(exc)


def predict(audio_file: str) -> tuple[dict[str, float], str, str]:
    if MODEL_LOAD_ERROR:
        raise gr.Error(MODEL_LOAD_ERROR)
    if not audio_file:
        raise gr.Error("Upload an audio file to continue.")

    audio = load_audio(audio_file)
    mel = to_mel(audio)
    tensor = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        logits = MODEL(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    top_index = int(np.argmax(probs))
    label = GENRES[top_index]
    ranking = {
        genre: float(prob)
        for genre, prob in sorted(
            zip(GENRES, probs),
            key=lambda item: item[1],
            reverse=True,
        )
    }
    summary = (
        f"Predicted genre: {label}\n\n"
        f"Top confidence: {probs[top_index]:.4f}\n"
        f"Model: EfficientNet-B0 spectrogram classifier"
    )
    return ranking, json.dumps(ranking, indent=2), summary


with gr.Blocks(title="Messy Mashup") as demo:
    gr.Markdown(
        """
        # Messy Mashup
        Upload a music clip and get a live genre prediction from the project deployment model.

        This demo serves a single EfficientNet checkpoint tuned for low-latency CPU inference.
        """
    )

    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Audio clip")

    with gr.Row():
        label_output = gr.Label(label="Genre ranking", num_top_classes=len(GENRES))
        json_output = gr.Textbox(label="Probability JSON", lines=12, show_copy_button=True)
        summary_output = gr.Textbox(label="Summary")

    submit = gr.Button("Predict")
    submit.click(
        fn=predict,
        inputs=[audio_input],
        outputs=[label_output, json_output, summary_output],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
    )
