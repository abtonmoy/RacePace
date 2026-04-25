"""`racepace extract-track` — generate a TrackMap JSON from a reference lap."""

from __future__ import annotations

from pathlib import Path

import typer

from racepace.features.laps import split_into_laps
from racepace.features.track_map import extract_track_map
from racepace.storage.session_store import SessionReader


def extract_track(
    db_path: Path = typer.Argument(..., help="Session DB written by `racepace record`."),
    lap: int = typer.Option(..., "--lap", help="Lap number to extract from (clean fast lap recommended)."),
    output: Path = typer.Option(..., "--output", help="Where to write the JSON track map."),
    session_id: str | None = typer.Option(None, "--session-id", help="Defaults to most recent."),
    track: str | None = typer.Option(None, "--track", help="Override the track name written into the map."),
    car: str | None = typer.Option(None, "--car", help="Override the car name."),
) -> None:
    """Extract corners from one lap and write a track-map JSON. Hand-edit names + notes after."""
    with SessionReader(db_path) as r:
        sid = session_id or r.latest_session_id()
        if sid is None:
            raise typer.BadParameter(f"No sessions in {db_path}.")
        info, frames = r.load_session(sid)

    laps = split_into_laps(frames)
    target = next((l for l in laps if l.lap_number == lap), None)
    if target is None or not target.frames:
        raise typer.BadParameter(f"Lap {lap} not found or empty.")

    tm = extract_track_map(
        target,
        track=track or info.track or "unknown",
        sim=info.sim,
        car=car or info.car,
    )
    tm.save(output)
    typer.echo(f"Wrote {len(tm.corners)} corners → {output}")
    typer.echo("Hand-edit the JSON to add corner names and notes.")
