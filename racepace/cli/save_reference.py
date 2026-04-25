"""`racepace save-reference` — save lap N as the reference lap for sim+track+car."""

from __future__ import annotations

from pathlib import Path

import typer

from racepace.features.laps import split_into_laps
from racepace.features.reference import reference_path, save_reference
from racepace.storage.session_store import SessionReader


def save_reference_cmd(
    db_path: Path = typer.Argument(..., help="Session DB."),
    lap: int = typer.Option(..., "--lap", help="Lap number to save as reference."),
    output: Path | None = typer.Option(
        None, "--output",
        help="Output Parquet path. Defaults to references/{sim}/{track}/{car}.parquet under the given root.",
    ),
    references_root: Path = typer.Option(
        Path("references"), "--references-root", help="Root dir for default-pathed references."
    ),
    session_id: str | None = typer.Option(None, "--session-id"),
    track: str | None = typer.Option(None, "--track"),
    car: str | None = typer.Option(None, "--car"),
) -> None:
    with SessionReader(db_path) as r:
        sid = session_id or r.latest_session_id()
        if sid is None:
            raise typer.BadParameter(f"No sessions in {db_path}.")
        info, frames = r.load_session(sid)

    laps = split_into_laps(frames)
    target = next((l for l in laps if l.lap_number == lap), None)
    if target is None or not target.frames:
        raise typer.BadParameter(f"Lap {lap} not found or empty.")

    final_track = track or info.track or "unknown"
    final_car = car or info.car
    out = output or reference_path(references_root, info.sim, final_track, final_car)
    save_reference(target, out, sim=info.sim, track=final_track, car=final_car)
    typer.echo(f"Saved reference for {info.sim}/{final_track}/{final_car or 'default'} → {out}")
