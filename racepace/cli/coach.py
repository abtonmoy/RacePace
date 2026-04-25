"""`racepace coach` — live coach + engineer + recording, all together.

Threads:
- Producer: adapter.stream_frames() → ringbuffer.push + writer.write_frame
- EngineerAgent: live strategic radio
- CoachAgent: fast (pace notes) + slow (sector coaching)
- Main: signal handler + post-session analyst pass on Ctrl-C
"""

from __future__ import annotations

import signal
import threading
from datetime import datetime, timezone
from pathlib import Path

import typer

from racepace.agents.coach import CoachAgent, CoachConfig
from racepace.agents.engineer import EngineerAgent
from racepace.comms.text_out import TextOutput
from racepace.features.reference import reference_path
from racepace.features.track_map import TrackMap
from racepace.storage.ringbuffer import RingBuffer
from racepace.storage.session_store import SessionWriter
from racepace.voice.cache import build_default_cache
from racepace.voice.player import Player
from racepace.voice.voice_out import VoiceOutput


def coach(
    sim: str = typer.Option(..., "--sim", help="Sim adapter to use: acc | f1 | simhub | mock."),
    output: Path = typer.Option(..., "--output", help="SQLite output path."),
    track_map_path: Path | None = typer.Option(
        None, "--track-map",
        help="Track map JSON. Defaults to racepace/tracks/{sim}/{track}.json.",
    ),
    references_root: Path = typer.Option(
        Path("references"), "--references-root",
        help="Root dir to search for a reference lap (sim/track/car).",
    ),
    voice: bool = typer.Option(False, "--voice/--no-voice", help="Enable TTS for engineer + slow-loop coaching."),
    voice_cache_dir: Path = typer.Option(
        Path("voice_cache"), "--voice-cache-dir", help="On-disk WAV cache for prebuilt clips and TTS memo."
    ),
    target_hz: float = typer.Option(30.0, "--hz"),
    buffer_seconds: float = typer.Option(60.0, "--buffer-seconds"),
    replay_path: Path | None = typer.Option(None, "--replay-path"),
    replay_speed: float = typer.Option(1.0, "--replay-speed"),
    run_analyst_on_finish: bool = typer.Option(True, "--analyst-on-finish/--no-analyst"),
) -> None:
    """Production command: drive the live pipeline. Ctrl-C triggers analyst."""
    from racepace.cli.record import _build_adapter

    adapter = _build_adapter(sim, target_hz, replay_path, replay_speed)
    ring = RingBuffer(capacity_seconds=buffer_seconds, expected_hz=target_hz)

    if voice:
        engineer_out = VoiceOutput(prefix="ENGINEER")
        coach_text_out = VoiceOutput(prefix="COACH", player=engineer_out.player, tts=engineer_out.tts)
    else:
        engineer_out = TextOutput(prefix="ENGINEER")
        coach_text_out = TextOutput(prefix="COACH")

    stop_event = threading.Event()

    def _on_sigint(signum, frame):
        if stop_event.is_set():
            raise KeyboardInterrupt
        stop_event.set()
        typer.echo("\nStopping...")

    signal.signal(signal.SIGINT, _on_sigint)

    with adapter:
        info = adapter.read_session_info()

        # Try to load a track map and reference for this sim+track+car
        tm_path = track_map_path or _default_track_map_path(info.sim, info.track)
        track_map = TrackMap.load(tm_path) if tm_path.exists() else None
        if track_map is None:
            typer.echo(f"  no track map at {tm_path} → coach runs in pace-notes-only mode (no maps).")
        ref = _try_load_reference(references_root, info.sim, info.track, info.car)
        if ref is None:
            typer.echo("  no reference lap → coach runs without comparative coaching.")

        # Build clip cache (silent if no TTS backend configured)
        clip_cache = build_default_cache(cache_dir=voice_cache_dir)
        player = Player()

        def _speak_clip(phrase: str) -> None:
            clip = clip_cache.get(phrase)
            if clip is None:
                return
            player.play(clip[0], clip[1], label=phrase)

        def _speak_text(msg: str) -> None:
            coach_text_out.send(msg)

        coach_agent = CoachAgent(
            ringbuffer=ring,
            session_info=info,
            track_map=track_map,
            clip_cache=clip_cache,
            speak_clip=_speak_clip,
            speak_text=_speak_text,
            reference=ref,
            config=CoachConfig(pace_notes_only=ref is None),
        )
        engineer_agent = EngineerAgent(ringbuffer=ring, session_info=info, output=engineer_out)

        with SessionWriter(output) as writer:
            writer.write_session(info)
            typer.echo(f"Recording session {info.session_id} → {output}")

            engineer_agent.start()
            coach_agent.start()

            written = 0
            last_lap = -1
            try:
                for frame in adapter.stream_frames():
                    ring.push(frame)
                    writer.write_frame(frame)
                    written += 1
                    if frame.lap_number != last_lap:
                        last_lap = frame.lap_number
                        typer.echo(f"  lap {last_lap}")
                    if stop_event.is_set():
                        break
            except KeyboardInterrupt:
                typer.echo("Force quit.")
            finally:
                coach_agent.stop()
                engineer_agent.stop()
                player.stop()
                writer.update_session_end(
                    ended_at=datetime.now(timezone.utc),
                    total_laps=max(last_lap, 0),
                )
    typer.echo(f"Wrote {written} frames across {max(last_lap, 0)} laps.")

    if run_analyst_on_finish and written > 0:
        try:
            from racepace.agents.analyst import analyze_session
            _, debrief = analyze_session(output)
            typer.echo("\n=== POST-SESSION DEBRIEF ===")
            typer.echo(debrief)
        except Exception as e:
            typer.echo(f"(analyst skipped: {e})")


def _default_track_map_path(sim: str, track: str | None) -> Path:
    base = Path(__file__).parent.parent / "tracks"
    return base / sim / f"{track or 'unknown'}.json"


def _try_load_reference(root: Path, sim: str, track: str | None, car: str | None):
    if not track:
        return None
    p = reference_path(root, sim, track, car)
    if not p.exists():
        return None
    try:
        from racepace.features.reference import load_reference
        return load_reference(p, sim=sim, track=track, car=car)
    except Exception:
        return None
