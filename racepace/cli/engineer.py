"""`racepace engineer` — live race engineer.

Always records to disk so Phase 1's analyst can review afterward.

Threads:
- Producer: adapter.stream_frames() → ringbuffer.push + writer.write_frame
- Consumer: EngineerAgent.run() → reads ringbuffer, speaks via TextOutput
- Main: signal handler, joins on stop
"""

from __future__ import annotations

import signal
import threading
from datetime import datetime, timezone
from pathlib import Path

import typer

from racepace.agents.engineer import EngineerAgent
from racepace.comms.text_out import TextOutput
from racepace.storage.ringbuffer import RingBuffer
from racepace.storage.session_store import SessionWriter


def engineer(
    sim: str = typer.Option(..., "--sim", help="Sim adapter to use: acc | mock."),
    output: Path = typer.Option(..., "--output", help="SQLite output path."),
    log_path: Path | None = typer.Option(
        None, "--log", help="Optional file to also write engineer messages to (one line each)."
    ),
    target_hz: float = typer.Option(30.0, "--hz", help="Target sample rate."),
    buffer_seconds: float = typer.Option(60.0, "--buffer-seconds", help="Ring buffer window."),
    replay_path: Path | None = typer.Option(None, "--replay-path", help="For --sim mock."),
    replay_speed: float = typer.Option(1.0, "--replay-speed", help="For --sim mock."),
) -> None:
    """Run the live race engineer until Ctrl-C."""
    from racepace.cli.record import _build_adapter

    adapter = _build_adapter(sim, target_hz, replay_path, replay_speed)
    ring = RingBuffer(capacity_seconds=buffer_seconds, expected_hz=target_hz)
    text_out = TextOutput(log_path=log_path, prefix="ENGINEER")

    stop_event = threading.Event()

    def _on_sigint(signum, frame):
        if stop_event.is_set():
            raise KeyboardInterrupt
        stop_event.set()
        typer.echo("\nStopping...")

    signal.signal(signal.SIGINT, _on_sigint)

    with adapter:
        info = adapter.read_session_info()
        with SessionWriter(output) as writer:
            writer.write_session(info)
            typer.echo(f"Recording session {info.session_id} → {output}")

            engineer_agent = EngineerAgent(ringbuffer=ring, session_info=info, output=text_out)
            agent_thread = engineer_agent.start()

            written = 0
            last_lap = -1
            try:
                for frame in adapter.stream_frames():
                    ring.push(frame)
                    writer.write_frame(frame)
                    written += 1
                    if frame.lap_number != last_lap:
                        last_lap = frame.lap_number
                        typer.echo(f"  lap {last_lap}  ({written} frames buffered)")
                    if stop_event.is_set():
                        break
            except KeyboardInterrupt:
                typer.echo("Force quit.")
            finally:
                engineer_agent.stop()
                writer.update_session_end(
                    ended_at=datetime.now(timezone.utc),
                    total_laps=max(last_lap, 0),
                )
    typer.echo(f"Wrote {written} frames across {max(last_lap, 0)} laps.")
