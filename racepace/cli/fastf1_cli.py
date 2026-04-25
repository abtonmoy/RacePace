"""`racepace fastf1-import` — pull a real F1 session and save it to a session DB."""

from __future__ import annotations

from pathlib import Path

import typer


def fastf1_import(
    year: int = typer.Option(..., "--year"),
    gp: str = typer.Option(..., "--gp", help="Grand Prix name or round number (e.g. 'Monza' or 14)."),
    session_type: str = typer.Option("R", "--session-type", help="R, Q, FP1, FP2, FP3, Sprint, etc."),
    driver: str = typer.Option(..., "--driver", help="3-letter code, e.g. VER, HAM, LEC."),
    output: Path = typer.Option(..., "--output"),
    lap_lo: int | None = typer.Option(None, "--lap-lo"),
    lap_hi: int | None = typer.Option(None, "--lap-hi"),
    cache_dir: Path = typer.Option(Path(".fastf1_cache"), "--cache-dir"),
) -> None:
    """Import a real F1 session into a RacePace session DB."""
    from racepace.data.fastf1_import import import_session

    lap_range: tuple[int, int] | None = None
    if lap_lo is not None and lap_hi is not None:
        lap_range = (lap_lo, lap_hi)

    typer.echo(f"Pulling {year} {gp} {session_type} for {driver}...")
    session_id, n_frames, n_laps = import_session(
        year=year,
        gp=gp,
        session_type=session_type,
        driver=driver,
        output_db=output,
        lap_range=lap_range,
        cache_dir=cache_dir,
    )
    typer.echo(f"Wrote session {session_id}: {n_frames} frames across {n_laps} laps → {output}")
