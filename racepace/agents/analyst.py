"""Post-session analyst.

One-shot summarizer (no tools, no agentic loop). Builds a structured
situation report from the feature extractors and asks Claude to write a
~300-500 word, three-section debrief: what went well, where time was
lost, what to work on next.

The LLM is bad at arithmetic. The prompt forbids inferring numbers; all
figures must come verbatim from the report dict.
"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from racepace.features.deg import DegProfile, tyre_degradation
from racepace.features.deltas import DeltaTrace, compare_laps
from racepace.features.laps import Lap, split_into_laps
from racepace.schema import SessionInfo, TelemetryFrame
from racepace.storage.session_store import SessionReader


_MODEL = "gemini-2.5-pro"

_SYSTEM_PROMPT = """You are a sim-racing post-session performance analyst. You receive a structured JSON situation report describing a single session (practice, qualifying, or race) and produce a written debrief for the driver.

Output exactly three Markdown sections, in this order:

## What went well
## Where time was lost
## What to work on next session

Constraints:
- Aim for 300-500 words total across all three sections.
- Refer to the metrics in the report verbatim. Do NOT estimate, average, or infer numerical values that are not present in the report.
- If a metric is null or missing, say "not reported" rather than guessing.
- Be specific. Cite lap numbers, sectors, and exact times from the report when relevant.
- Be honest. If the data shows poor consistency or heavy degradation, say so.
- Do not invent events that are not in the `notable_events` field.
- No preamble, no closing pleasantries — start with the first heading.
"""


def analyze_session(
    db_path: str | Path,
    session_id: str | None = None,
    api_key: str | None = None,
    save_to: str | Path | None = None,
) -> tuple[dict, str]:
    """Run the full pipeline. Returns (report_dict, debrief_markdown)."""
    with SessionReader(db_path) as reader:
        sid = session_id or reader.latest_session_id()
        if sid is None:
            raise RuntimeError(f"No sessions in {db_path}.")
        info, frames = reader.load_session(sid)

    laps = split_into_laps(frames)
    report = build_report(info, frames, laps)

    debrief = _call_llm(report, api_key=api_key)

    if save_to is not None:
        Path(save_to).write_text(debrief, encoding="utf-8")

    return report, debrief


def build_report(
    info: SessionInfo,
    frames: list[TelemetryFrame],
    laps: list[Lap],
) -> dict[str, Any]:
    """The structured situation report. This is the artifact the LLM sees."""
    deg = tyre_degradation(laps)

    complete = [l for l in laps if l.is_complete and l.lap_time_s is not None]
    clean = [l for l in complete if l.is_clean is not False]

    best_lap = min(complete, key=lambda l: l.lap_time_s) if complete else None  # type: ignore[arg-type]

    lap_times = [l.lap_time_s for l in complete if l.lap_time_s is not None]
    consistency = _consistency(lap_times)

    fuel_per_lap = [l.fuel_used_kg for l in complete if l.fuel_used_kg is not None]
    avg_fuel_per_lap = sum(fuel_per_lap) / len(fuel_per_lap) if fuel_per_lap else None

    sector_focus = _sector_focus(best_lap, [l for l in complete if l is not best_lap])

    notable = _notable_events(laps)

    return {
        "session": {
            "sim": info.sim,
            "track": info.track,
            "car": info.car,
            "session_type": info.session_type,
            "started_at": info.started_at.isoformat() if info.started_at else None,
            "ended_at": info.ended_at.isoformat() if info.ended_at else None,
            "weather": info.weather,
            "track_temp_c": info.track_temp_c,
            "air_temp_c": info.air_temp_c,
            "frame_count": len(frames),
        },
        "lap_summary": {
            "total_laps": len(laps),
            "completed_laps": len(complete),
            "clean_laps": len(clean),
            "dirty_laps": sum(1 for l in laps if l.is_clean is False),
            "best_lap": _lap_summary(best_lap),
            "average_lap_time_s": (sum(lap_times) / len(lap_times)) if lap_times else None,
            "consistency_stdev_s": consistency,
        },
        "fuel": {
            "avg_fuel_used_per_lap_kg": avg_fuel_per_lap,
            "starting_fuel_kg": _first_value(frames, "fuel_kg"),
            "ending_fuel_kg": _last_value(frames, "fuel_kg"),
        },
        "tyre_degradation": {
            "stints": [_stint_dict(s) for s in deg.stints],
        },
        "sector_focus": sector_focus,
        "notable_events": notable,
    }


def _lap_summary(lap: Lap | None) -> dict | None:
    if lap is None:
        return None
    return {
        "lap_number": lap.lap_number,
        "lap_time_s": lap.lap_time_s,
        "avg_speed_kph": lap.avg_speed_kph,
        "max_speed_kph": lap.max_speed_kph,
        "is_clean": lap.is_clean,
        "compound": lap.compound,
        "fuel_used_kg": lap.fuel_used_kg,
        "tyre_temp_avg_c": lap.tyre_temp_avg_c,
    }


def _stint_dict(stint) -> dict:
    d = asdict(stint) if is_dataclass(stint) else dict(stint)
    return {
        "stint_index": d["stint_index"],
        "compound": d.get("compound"),
        "laps": d.get("laps"),
        "lap_numbers": d.get("lap_numbers"),
        "fit_lap_numbers": d.get("fit_lap_numbers"),
        "slope_s_per_lap": d.get("slope_s_per_lap"),
        "slope_ci95_s": d.get("slope_ci95_s"),
        "intercept_s": d.get("intercept_s"),
        "r_squared": d.get("r_squared"),
        "tyre_temp_drift_c": d.get("tyre_temp_drift_c"),
        "out_of_window_pct": d.get("out_of_window_pct"),
    }


def _consistency(lap_times: list[float]) -> float | None:
    if len(lap_times) < 2:
        return None
    return statistics.stdev(lap_times)


def _sector_focus(best: Lap | None, others: list[Lap]) -> dict | None:
    """Identify the sector where the most time is lost vs the best lap.

    Compares each non-best lap against the best lap by track distance,
    bins the cumulative time delta into 3 sectors of equal length, and
    averages each sector's loss across laps. Reports the worst sector.
    """
    if best is None or not others:
        return None

    sector_losses_avg = [0.0, 0.0, 0.0]
    counted = 0
    for lap in others:
        delta = compare_laps(best, lap)
        per_sector = _bin_delta_by_sector(delta)
        if per_sector is None:
            continue
        for i in range(3):
            sector_losses_avg[i] += per_sector[i]
        counted += 1

    if counted == 0:
        return None
    sector_losses_avg = [v / counted for v in sector_losses_avg]
    worst_idx = max(range(3), key=lambda i: sector_losses_avg[i])

    return {
        "reference_lap_number": best.lap_number,
        "compared_lap_count": counted,
        "avg_time_loss_per_sector_s": {
            "s1": sector_losses_avg[0],
            "s2": sector_losses_avg[1],
            "s3": sector_losses_avg[2],
        },
        "worst_sector": worst_idx + 1,
        "worst_sector_avg_loss_s": sector_losses_avg[worst_idx],
    }


def _bin_delta_by_sector(delta: DeltaTrace) -> list[float] | None:
    cum = delta.cumulative_time_delta_s
    xs = delta.distance_m
    if not cum or not xs:
        return None
    n = len(xs)
    third = n // 3
    if third < 2:
        return None
    # Sector loss = (cum at end) - (cum at start) for each third.
    losses = []
    for i in range(3):
        start = i * third
        end = (i + 1) * third if i < 2 else n - 1
        if cum[start] is None or cum[end] is None:
            return None
        losses.append(cum[end] - cum[start])
    return losses


def _notable_events(laps: list[Lap]) -> list[dict]:
    """Off-tracks (dirty laps), implausibly slow laps, and compound changes."""
    events: list[dict] = []
    median_time = None
    times = [l.lap_time_s for l in laps if l.lap_time_s and l.is_clean]
    if times:
        median_time = statistics.median(times)

    prev_compound: str | None = None
    for lap in laps:
        if lap.is_clean is False:
            events.append({
                "type": "off_track_or_spin",
                "lap_number": lap.lap_number,
                "detail": "Sample-to-sample speed drop > 30 kph at racing speed.",
            })
        if (
            median_time
            and lap.lap_time_s
            and lap.lap_time_s > median_time * 1.10
            and lap.is_complete
        ):
            events.append({
                "type": "slow_lap",
                "lap_number": lap.lap_number,
                "lap_time_s": lap.lap_time_s,
                "median_clean_lap_time_s": median_time,
            })
        if prev_compound and lap.compound and lap.compound != prev_compound:
            events.append({
                "type": "compound_change",
                "lap_number": lap.lap_number,
                "from": prev_compound,
                "to": lap.compound,
            })
        if lap.compound:
            prev_compound = lap.compound
    return events


def _first_value(frames: list[TelemetryFrame], attr: str):
    for f in frames:
        v = getattr(f, attr, None)
        if v is not None:
            return v
    return None


def _last_value(frames: list[TelemetryFrame], attr: str):
    for f in reversed(frames):
        v = getattr(f, attr, None)
        if v is not None:
            return v
    return None


def _call_llm(report: dict, api_key: str | None = None) -> str:
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "Set GEMINI_API_KEY (or GOOGLE_API_KEY), or pass api_key explicitly."
        )
    client = genai.Client(api_key=key)

    user_text = (
        "Situation report (JSON):\n\n```json\n"
        + json.dumps(report, indent=2, default=str)
        + "\n```\n\nWrite the debrief now."
    )

    response = client.models.generate_content(
        model=_MODEL,
        contents=user_text,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            max_output_tokens=4000,
        ),
    )

    text = response.text or ""
    return text.strip()
