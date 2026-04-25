"""`racepace openf1-import` — pull a real F1 session from openf1.org."""

from __future__ import annotations

from pathlib import Path

import typer


def openf1_import(
    year: int = typer.Option(..., "--year"),
    country: str = typer.Option(..., "--country", help="Country / location / circuit name (matched case-insensitively)."),
    session_name: str = typer.Option("Race", "--session-name", help="e.g. Race, Qualifying, Practice 1."),
    driver_number: int = typer.Option(..., "--driver-number", help="Numeric driver number, e.g. 1=VER, 44=HAM."),
    output: Path = typer.Option(..., "--output"),
    lap_lo: int | None = typer.Option(None, "--lap-lo"),
    lap_hi: int | None = typer.Option(None, "--lap-hi"),
) -> None:
    """Import a real F1 session from openf1.org into a RacePace session DB."""
    from racepace.data.openf1_import import import_session

    lap_range = (lap_lo, lap_hi) if (lap_lo is not None and lap_hi is not None) else None

    typer.echo(f"Pulling {year} {country} {session_name} for driver #{driver_number}...")
    session_id, n_frames, n_laps = import_session(
        year=year,
        country_or_circuit=country,
        session_name=session_name,
        driver_number=driver_number,
        output_db=output,
        lap_range=lap_range,
    )
    typer.echo(f"Wrote session {session_id}: {n_frames} frames across {n_laps} laps → {output}")
