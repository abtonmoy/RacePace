"""Top-level Typer app: racepace record / analyze / engineer / coach / extract-track / save-reference."""

from __future__ import annotations

import typer

from racepace.cli import analyze as _analyze
from racepace.cli import coach as _coach
from racepace.cli import engineer as _engineer
from racepace.cli import extract_track as _extract_track
from racepace.cli import fastf1_cli as _fastf1
from racepace.cli import openf1_cli as _openf1
from racepace.cli import record as _record
from racepace.cli import save_reference as _save_reference

app = typer.Typer(
    add_completion=False,
    help="RacePace — record sim telemetry, run a live engineer + coach, debrief afterward.",
    no_args_is_help=True,
)
app.command("record")(_record.record)
app.command("analyze")(_analyze.analyze)
app.command("engineer")(_engineer.engineer)
app.command("coach")(_coach.coach)
app.command("extract-track")(_extract_track.extract_track)
app.command("save-reference")(_save_reference.save_reference_cmd)
app.command("fastf1-import")(_fastf1.fastf1_import)
app.command("openf1-import")(_openf1.openf1_import)


if __name__ == "__main__":
    app()
