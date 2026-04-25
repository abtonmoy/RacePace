"""`racepace analyze` — run the analyst on a session DB and print/save the debrief."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from racepace.agents.analyst import analyze_session


def analyze(
    db_path: Path = typer.Argument(..., help="Path to a session DB written by `racepace record`."),
    session_id: str | None = typer.Option(
        None, "--session-id", help="Session UUID. Defaults to the most recent in the DB."
    ),
    save_to: Path | None = typer.Option(
        None, "--save", help="Write the Markdown debrief to this path."
    ),
    print_report: bool = typer.Option(
        False, "--print-report", help="Also print the structured situation report (JSON)."
    ),
    api_key: str | None = typer.Option(
        None, "--api-key", help="Override GEMINI_API_KEY / GOOGLE_API_KEY."
    ),
) -> None:
    """Build the report and the debrief; print to stdout."""
    report, debrief = analyze_session(
        db_path=db_path,
        session_id=session_id,
        api_key=api_key,
        save_to=save_to,
    )
    if print_report:
        typer.echo("=== situation report ===")
        typer.echo(json.dumps(report, indent=2, default=str))
        typer.echo("=== debrief ===")
    typer.echo(debrief)
