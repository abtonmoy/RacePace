"""Engineer trigger logic and no-chatter test.

We drive the engineer synchronously by directly invoking the public hooks
(`build_report` → `should_speak` → `generate_message`) instead of starting
the thread. That gives deterministic event ordering and no real time.

The LLM is mocked: it echoes the trigger so we can grep for which trigger
caused each spoken message.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Iterable

from racepace.agents.engineer import EngineerAgent
from racepace.comms.text_out import CapturingOutput
from racepace.schema import SessionInfo, TelemetryFrame
from racepace.storage.ringbuffer import RingBuffer


def _info(track="fixture-track", total_laps=30):
    return SessionInfo(
        session_id="test-engineer-session",
        sim="acc",
        started_at=datetime(2026, 4, 25, tzinfo=timezone.utc),
        track=track,
        car="fixture-car",
        session_type="race",
        total_laps=total_laps,
    )


def _build_lap_frames(
    *,
    lap_number: int,
    start_t: float,
    lap_time_s: float = 90.0,
    fuel_kg: float,
    fuel_used_kg: float,
    wear_pct_start: float,
    wear_per_lap_pct: float,
    last_lap_time_s: float | None = None,
    best_lap_time_s: float | None = None,
    position: int = 1,
    gap_behind_s: float | None = None,
    hz: float = 5.0,
) -> Iterable[TelemetryFrame]:
    n_frames = int(lap_time_s * hz)
    fuel_step = fuel_used_kg / n_frames
    wear_step = wear_per_lap_pct / n_frames
    for i in range(n_frames):
        progress = i / n_frames
        wear = wear_pct_start + wear_step * i
        yield TelemetryFrame(
            timestamp_s=start_t + i / hz,
            lap_number=lap_number,
            lap_distance_m=progress * 4000.0,
            lap_distance_pct=progress,
            speed_kph=170.0,
            throttle_pct=80.0,
            brake_pct=0.0,
            gear=4,
            rpm=7000.0,
            fuel_kg=fuel_kg - fuel_step * i,
            tyre_wear_pct={"fl": wear, "fr": wear, "rl": wear, "rr": wear},
            tyre_temp_c={"fl": 80.0, "fr": 80.0, "rl": 78.0, "rr": 78.0},
            tyre_compound="dry",
            position=position,
            gap_behind_s=gap_behind_s,
            last_lap_time_s=last_lap_time_s,
            best_lap_time_s=best_lap_time_s,
            current_sector=int(progress * 3),
        )


def _drive(
    agent: EngineerAgent,
    ring: RingBuffer,
    frames: Iterable[TelemetryFrame],
    tick_every_s: float = 2.0,
) -> list[tuple[str, str]]:
    """Push frames into the buffer and tick the engineer at `tick_every_s`.

    Returns list of (trigger, message) tuples for inspection.
    """
    spoken: list[tuple[str, str]] = []
    next_tick_t = 0.0
    last_t: float | None = None
    for f in frames:
        ring.push(f)
        last_t = f.timestamp_s
        if f.timestamp_s >= next_tick_t:
            report = agent.build_report()
            if report is not None:
                yes, trigger = agent.should_speak(report)
                if yes:
                    msg = agent.generate_message(report, trigger or "")
                    if msg:
                        spoken.append((trigger or "", msg))
            next_tick_t = f.timestamp_s + tick_every_s
    return spoken


def _mock_llm():
    return lambda trigger, _report_json: f"[{trigger}] mock message"


# --- no-chatter test ----------------------------------------------------------

def test_no_chatter_in_uneventful_30_lap_stint():
    """Constant pace, plenty of fuel and tyres: engineer should be near-silent."""
    info = _info(total_laps=200)  # plenty of room remaining
    ring = RingBuffer(capacity_seconds=120, expected_hz=5)
    agent = EngineerAgent(
        ringbuffer=ring,
        session_info=info,
        output=CapturingOutput(),
        llm_call=_mock_llm(),
    )

    frames: list[TelemetryFrame] = []
    t = 0.0
    fuel = 200.0
    last_t = 0.0
    best = 90.0
    for lap in range(1, 31):
        last_lap_time = 90.0 if lap > 1 else None
        chunk = list(_build_lap_frames(
            lap_number=lap,
            start_t=t,
            lap_time_s=90.0,
            fuel_kg=fuel,
            fuel_used_kg=1.0,  # very economical, won't trigger fuel warnings
            wear_pct_start=(lap - 1) * 0.5,  # 0.5%/lap, never reaches critical
            wear_per_lap_pct=0.5,
            last_lap_time_s=last_lap_time,
            best_lap_time_s=best,
        ))
        frames.extend(chunk)
        fuel -= 1.0
        t += 90.0

    spoken = _drive(agent, ring, frames, tick_every_s=2.0)
    # Periodic message every 5 laps from lap ~5 onwards → ≤6 periodic messages.
    assert len(spoken) < 10, f"engineer was too chatty: {[t for t, _ in spoken]}"


# --- pit window trigger -------------------------------------------------------

def test_pit_window_opens_when_fuel_runs_low():
    info = _info(total_laps=20)
    ring = RingBuffer(capacity_seconds=120, expected_hz=5)
    agent = EngineerAgent(
        ringbuffer=ring,
        session_info=info,
        output=CapturingOutput(),
        llm_call=_mock_llm(),
    )

    # Start with only 12 kg fuel, 2.5 kg/lap → 4.8 laps remaining; race is 20.
    # That immediately puts us in fuel-margin trouble.
    frames: list[TelemetryFrame] = []
    t = 0.0
    fuel = 12.0
    best = 90.0
    for lap in range(1, 8):
        last_lap_time = 90.0 if lap > 1 else None
        chunk = list(_build_lap_frames(
            lap_number=lap,
            start_t=t,
            lap_time_s=90.0,
            fuel_kg=fuel,
            fuel_used_kg=2.5,
            wear_pct_start=(lap - 1) * 1.0,
            wear_per_lap_pct=1.0,
            last_lap_time_s=last_lap_time,
            best_lap_time_s=best,
        ))
        frames.extend(chunk)
        fuel -= 2.5
        t += 90.0

    spoken = _drive(agent, ring, frames, tick_every_s=2.0)
    triggers = {t for t, _ in spoken}
    assert "pit_window_open" in triggers, f"expected pit_window_open in {triggers}"


def test_position_change_triggers_one_message():
    info = _info(total_laps=10)  # short race + lots of fuel = no fuel/pit triggers
    ring = RingBuffer(capacity_seconds=120, expected_hz=5)
    agent = EngineerAgent(
        ringbuffer=ring,
        session_info=info,
        output=CapturingOutput(),
        llm_call=_mock_llm(),
    )

    frames: list[TelemetryFrame] = []
    t = 0.0
    fuel = 100.0  # ~100 laps of fuel for a 10-lap race
    for lap in range(1, 6):
        last_lap_time = 90.0 if lap > 1 else None
        # Position changes from 5 → 3 mid-stint
        position = 5 if lap < 3 else 3
        chunk = list(_build_lap_frames(
            lap_number=lap,
            start_t=t,
            lap_time_s=90.0,
            fuel_kg=fuel,
            fuel_used_kg=1.0,
            wear_pct_start=(lap - 1) * 0.5,
            wear_per_lap_pct=0.5,
            last_lap_time_s=last_lap_time,
            best_lap_time_s=90.0,
            position=position,
        ))
        frames.extend(chunk)
        fuel -= 1.0
        t += 90.0

    spoken = _drive(agent, ring, frames, tick_every_s=2.0)
    pos_triggers = [t for t, _ in spoken if t.startswith("position_change_")]
    assert len(pos_triggers) >= 1, f"expected at least one position_change_, got {[t for t, _ in spoken]}"


def test_engineer_does_not_speak_with_empty_buffer():
    info = _info()
    ring = RingBuffer(capacity_seconds=10, expected_hz=5)
    agent = EngineerAgent(
        ringbuffer=ring,
        session_info=info,
        output=CapturingOutput(),
        llm_call=_mock_llm(),
    )
    assert agent.build_report() is None
