"""Runtime diagnostic for the audio + TTS stack.

NOT a pytest test — it actually plays sound, requires PortAudio + Piper to be
installed, and depends on the host's audio device. Pytest's default
collector skips it because the filename lacks the `test_` prefix.

Run from the repo root:
    PIPER_MODEL_PATH=~/piper-voices/en_US-amy-medium.onnx \\
      uv run python -m racepace.tests.voice_test

Optional flags:
    --phrase 'something to say'
    --skip-audio                 # skip the actual playback
"""

from __future__ import annotations

import os
import shutil

import typer

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    phrase: str = typer.Option(
        "this is racepace, voice output is working",
        "--phrase",
        help="What to synthesize and play.",
    ),
    skip_audio: bool = typer.Option(
        False, "--skip-audio", help="Skip the actual playback step (just synthesize)."
    ),
) -> None:
    """Diagnose the voice output stack and play one test phrase."""
    typer.echo("=== audio backend probe ===")
    try:
        import sounddevice as sd
        try:
            default_out = sd.default.device[1]
            typer.echo(f"  sounddevice OK; default output device idx = {default_out}")
            for i, d in enumerate(sd.query_devices()):
                if d["max_output_channels"] > 0:
                    marker = "*" if i == default_out else " "
                    typer.echo(f"    {marker} [{i}] {d['name']} ({d['max_output_channels']} out @ {int(d['default_samplerate'])} Hz)")
        except Exception as e:
            typer.echo(f"  sounddevice imported but device query failed: {e}")
    except ImportError:
        typer.echo("  sounddevice NOT installed.")
        typer.echo("    fix: uv sync --extra voice")
    except OSError as e:
        typer.echo(f"  sounddevice present but PortAudio unavailable: {e}")
        typer.echo("    fix: install your system PortAudio package (Ubuntu: apt install libportaudio2)")

    typer.echo("")
    typer.echo("=== TTS backend probe ===")
    piper_bin = shutil.which("piper")
    piper_model = os.environ.get("PIPER_MODEL_PATH")
    typer.echo(f"  piper on PATH: {piper_bin or 'NO'}")
    typer.echo(f"  PIPER_MODEL_PATH: {piper_model or '(unset)'}")
    if piper_model and not os.path.exists(piper_model):
        typer.echo(f"  WARNING: PIPER_MODEL_PATH points to a non-existent file")

    if not piper_bin or not piper_model:
        typer.echo("")
        typer.echo("  To enable real speech:")
        typer.echo("    1. uv pip install piper-tts                       # Python wheel + binary")
        typer.echo("       OR install Piper from https://github.com/rhasspy/piper/releases")
        typer.echo("    2. Download a voice model (.onnx + .onnx.json):")
        typer.echo("       https://github.com/rhasspy/piper/blob/master/VOICES.md")
        typer.echo("    3. export PIPER_MODEL_PATH=/abs/path/to/voice.onnx")
        typer.echo("")
        typer.echo("  Falling back to NullTTSBackend (silent buffers) for the synth test below.")

    typer.echo("")
    typer.echo("=== synth + play ===")
    from racepace.voice.live_tts import default_tts
    from racepace.voice.player import NullBackend, Player, default_backend

    tts = default_tts()
    typer.echo(f"  TTS backend selected: {type(tts).__name__} (voice_id={tts.voice_id})")
    try:
        samples, sr = tts.synthesize(phrase)
        typer.echo(f"  synthesized OK: {len(samples)} samples @ {sr} Hz ({len(samples)/sr:.2f}s)")
    except Exception as e:
        typer.echo(f"  synthesize FAILED: {e}")
        raise typer.Exit(1)

    if skip_audio:
        typer.echo("  --skip-audio: not playing")
        return

    backend = default_backend()
    typer.echo(f"  audio backend selected: {type(backend).__name__}")
    if isinstance(backend, NullBackend):
        typer.echo("  (NullBackend = no actual sound; install --extra voice for real playback)")
    player = Player(backend=backend, max_age_s=5.0)
    typer.echo(f"  playing {phrase!r}...")
    player.play(samples, sr, label="voice-test")
    import time
    time.sleep(max(1.0, len(samples) / sr + 0.3))
    player.stop()
    typer.echo("  done.")


if __name__ == "__main__":
    app()
