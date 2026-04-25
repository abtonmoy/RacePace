"""`racepace record` — connect adapter, stream frames into storage."""

from __future__ import annotations

import signal
from datetime import datetime, timezone
from pathlib import Path

import typer

from racepace.adapters.base import AbstractAdapter
from racepace.storage.session_store import SessionWriter


def record(
    sim: str = typer.Option(..., "--sim", help="Sim adapter to use: acc | mock."),
    output: Path = typer.Option(..., "--output", help="SQLite output path."),
    target_hz: float = typer.Option(30.0, "--hz", help="Target sample rate."),
    replay_path: Path | None = typer.Option(
        None, "--replay-path", help="For --sim mock: path to a recorded session DB."
    ),
    replay_speed: float = typer.Option(
        1.0, "--replay-speed", help="For --sim mock: 1.0 = real time, 0.0 = as fast as possible."
    ),
) -> None:
    """Stream frames from a sim into a session DB until Ctrl-C or session ends."""
    adapter = _build_adapter(sim, target_hz, replay_path, replay_speed)

    stop = {"flag": False}

    def _on_sigint(signum, frame):
        if stop["flag"]:
            raise KeyboardInterrupt
        stop["flag"] = True
        typer.echo("\nStopping after current frame...")

    signal.signal(signal.SIGINT, _on_sigint)

    written = 0
    last_lap = -1
    with adapter:
        info = adapter.read_session_info()
        with SessionWriter(output) as writer:
            writer.write_session(info)
            typer.echo(f"Recording session {info.session_id} → {output}")
            try:
                for frame in adapter.stream_frames():
                    writer.write_frame(frame)
                    written += 1
                    if frame.lap_number != last_lap:
                        last_lap = frame.lap_number
                        typer.echo(f"  lap {last_lap}  ({written} frames)")
                    if stop["flag"]:
                        break
            except KeyboardInterrupt:
                typer.echo("Force quit.")
            finally:
                writer.update_session_end(
                    ended_at=datetime.now(timezone.utc),
                    total_laps=max(last_lap, 0),
                )
    typer.echo(f"Wrote {written} frames across {max(last_lap, 0)} laps.")


def _build_adapter(
    sim: str,
    target_hz: float,
    replay_path: Path | None,
    replay_speed: float,
    f1_port: int = 20777,
    simhub_url: str = "ws://127.0.0.1:8888/",
) -> AbstractAdapter:
    sim = sim.lower()
    if sim == "acc":
        from racepace.adapters.acc import AccAdapter
        return AccAdapter(target_hz=target_hz)
    if sim == "f1":
        from racepace.adapters.f1 import F1Adapter
        return F1Adapter(port=f1_port, target_hz=target_hz)
    if sim == "simhub":
        from racepace.adapters.simhub import SimHubAdapter
        return SimHubAdapter(url=simhub_url, target_hz=target_hz)
    if sim == "mock":
        if replay_path is None:
            raise typer.BadParameter("--replay-path is required when --sim mock.")
        from racepace.adapters.mock import MockAdapter
        return MockAdapter(replay_path, speed=replay_speed)
    raise typer.BadParameter(f"Unknown sim: {sim!r}. Supported: acc, f1, simhub, mock.")
