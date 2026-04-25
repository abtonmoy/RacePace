"""Text-to-speech for the slow loop and the engineer.

Backends:
- `PiperTTSBackend` — calls the local `piper` binary (offline, free,
  reasonable quality). Requires `piper-tts` installed and a voice model.
- `NullTTSBackend` — returns a short silent buffer. Used when no TTS
  backend is configured; pipeline still runs end-to-end.
- `RecordingTTSBackend` — captures requested texts; used by tests.

`OnDiskCachedTTS` wraps any backend to memoize on a SHA1 of the text +
voice id. Cuts both latency and cost when the same sentence repeats.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import struct
import subprocess
import tempfile
import wave
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


def _silent(duration_s: float = 0.2, sample_rate: int = 22050) -> tuple[np.ndarray, int]:
    return np.zeros(int(duration_s * sample_rate), dtype=np.float32), sample_rate


class TTSBackend(ABC):
    voice_id: str = "default"

    @abstractmethod
    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Return (samples, sample_rate). May block."""


class NullTTSBackend(TTSBackend):
    voice_id = "null"

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        # 100ms of silence per word — gives the player a realistic clip length.
        words = max(1, len(text.split()))
        return _silent(duration_s=0.12 * words)


class RecordingTTSBackend(TTSBackend):
    """Captures texts requested. For tests."""
    voice_id = "recording"

    def __init__(self) -> None:
        self.requested: list[str] = []

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        self.requested.append(text)
        return _silent(duration_s=0.05)


class PiperTTSBackend(TTSBackend):
    """Subprocess to a local `piper` binary.

    Requires `piper` on PATH and a `.onnx` voice model.

    Tested invocation:
        echo "hello" | piper --model en_US-amy-medium.onnx --output_file out.wav
    """

    def __init__(self, model_path: str | Path, piper_bin: str = "piper") -> None:
        if not shutil.which(piper_bin):
            raise RuntimeError(f"{piper_bin!r} not found on PATH")
        self.piper_bin = piper_bin
        self.model_path = str(Path(model_path).expanduser().resolve())
        self.voice_id = Path(self.model_path).stem

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            out_path = tf.name
        try:
            subprocess.run(
                [self.piper_bin, "--model", self.model_path, "--output_file", out_path],
                input=text.encode("utf-8"),
                check=True,
                capture_output=True,
                timeout=30,
            )
            return _read_wav_mono(out_path)
        finally:
            try:
                os.unlink(out_path)
            except OSError:
                pass


class OnDiskCachedTTS(TTSBackend):
    """Decorator that memoizes synthesize() output to a directory."""

    def __init__(self, inner: TTSBackend, cache_dir: str | Path) -> None:
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.voice_id = inner.voice_id

    def _key(self, text: str) -> Path:
        h = hashlib.sha1(f"{self.voice_id}::{text}".encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{h}.wav"

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        path = self._key(text)
        if path.exists():
            try:
                return _read_wav_mono(path)
            except Exception:
                pass  # corrupt cache — fall through and re-synthesize
        samples, sr = self.inner.synthesize(text)
        try:
            _write_wav_mono(path, samples, sr)
        except Exception:
            pass
        return samples, sr


# --- WAV IO ------------------------------------------------------------------

def _read_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        n_channels = w.getnchannels()
        sample_rate = w.getframerate()
        sample_width = w.getsampwidth()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)
    if sample_width == 2:
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sample_width == 1:
        arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")
    if n_channels > 1:
        arr = arr.reshape(-1, n_channels).mean(axis=1)
    return arr, sample_rate


def _write_wav_mono(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767.0).astype(np.int16).tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)


def default_tts() -> TTSBackend:
    """Pick PiperTTSBackend if a model path env var is set, else NullTTSBackend."""
    model = os.environ.get("PIPER_MODEL_PATH")
    if model and Path(model).exists():
        try:
            return PiperTTSBackend(model)
        except Exception:
            pass
    return NullTTSBackend()
