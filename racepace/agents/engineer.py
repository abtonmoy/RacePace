"""Race engineer agent.

Tick interval: 2s. Default model: gemini-2.5-flash (cheap + fast; quality
is plenty for one-or-two-sentence radio calls). The trigger state machine
in `should_speak` is doing the load-bearing work — the LLM only ever sees
events the gatekeeper already decided are worth speaking on.

The engineer never invents numbers. The prompt forbids it; the situation
report carries every figure the model is allowed to quote.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Callable

from google import genai
from google.genai import types

from racepace.agents.base import LiveAgent
from racepace.features.situation import (
    SituationReport,
    StrategyState,
    build_situation,
)
from racepace.schema import SessionInfo
from racepace.storage.ringbuffer import RingBuffer


_MODEL = "gemini-2.5-flash"

_SYSTEM_PROMPT = """You are a calm, terse sim-racing race engineer on the pit wall. You speak to the driver over the radio between corners.

Style:
- One or two short sentences. No preamble. No closing pleasantries.
- Quote numbers verbatim from the situation report — never estimate or compute.
- Round numbers shown to one decimal place when reading them aloud (e.g. "fuel margin 1.2 laps").
- Use the driver's first name only if it appears in the report. Otherwise no salutation.
- Match the trigger: if the trigger is "pit window open", speak about pitting. Don't free-associate.

What to say for each trigger:
- pit_window_open: tell them the window is open and the optimal lap if known
- fuel_low: state fuel margin in laps, recommend lift-and-coast if margin < 1 lap
- tyre_critical: state laps until critical, suggest box if < 3 laps
- undercut_threat: name the gap behind, recommend a defensive pit
- overtake_opportunity: name the gap ahead, encourage the move
- position_change: brief acknowledgment ("P3 now") — no embellishment
- rain_incoming: name the trigger, recommend boxing for wets if certain
- periodic_status: pace + tyres + fuel in one sentence

Refuse to speak if you have no useful information. Output the literal token NO_COMMENT and nothing else if the trigger is not actionable from the report."""


# Trigger thresholds — tweak per fleet of drivers later.
FUEL_MARGIN_WARN_LAPS = 1.0
FUEL_MARGIN_CRITICAL_LAPS = 0.5
TYRE_CRITICAL_PCT = 80.0
PERIODIC_LAP_INTERVAL = 5
SUPPRESSION_S = 15.0


class EngineerAgent(LiveAgent):
    tick_interval_s = 2.0

    def __init__(
        self,
        ringbuffer: RingBuffer,
        session_info: SessionInfo,
        output,
        llm_call: Callable[[str, str], str] | None = None,
        api_key: str | None = None,
    ) -> None:
        super().__init__(ringbuffer, session_info, output)
        self.strategy_state = StrategyState()
        self._llm_call = llm_call or _default_llm_call(api_key)

        # Per-trigger dedupe state
        self._announced_pit_window = False
        self._announced_rain = False
        self._announced_fuel_warn = False
        self._announced_fuel_critical = False
        self._announced_tyre_critical = False
        self._undercut_active = False
        self._last_position: int | None = None
        self._last_periodic_lap: int | None = None
        # Suppression clock is the *session* clock (latest frame timestamp_s),
        # not wall-clock. Tests run faster than real time and the live system
        # only cares about race time anyway.
        self._last_message_session_t: float = float("-inf")
        self._latest_session_t: float = 0.0

    # --- LiveAgent contract ----------------------------------------------------

    def build_report(self) -> SituationReport | None:
        snapshot = self.ringbuffer.snapshot()
        if snapshot:
            self._latest_session_t = snapshot[-1].timestamp_s
        return build_situation(self.session_info, snapshot, self.strategy_state)

    def should_speak(self, report: SituationReport) -> tuple[bool, str | None]:
        suppressed = (self._latest_session_t - self._last_message_session_t) < SUPPRESSION_S

        # Critical first-time events bypass suppression.
        if report.fuel_margin_laps is not None and report.fuel_margin_laps < FUEL_MARGIN_CRITICAL_LAPS:
            if not self._announced_fuel_critical:
                self._announced_fuel_critical = True
                return True, "fuel_critical"

        if report.avg_tyre_wear_pct is not None and report.avg_tyre_wear_pct >= TYRE_CRITICAL_PCT:
            if not self._announced_tyre_critical:
                self._announced_tyre_critical = True
                return True, "tyre_critical"

        if suppressed:
            return False, None

        # First-time strategic events
        if report.pit_window_open and not self._announced_pit_window:
            self._announced_pit_window = True
            return True, "pit_window_open"

        if report.rain_incoming and not self._announced_rain:
            self._announced_rain = True
            return True, "rain_incoming"

        if report.fuel_margin_laps is not None and report.fuel_margin_laps < FUEL_MARGIN_WARN_LAPS:
            if not self._announced_fuel_warn:
                self._announced_fuel_warn = True
                return True, "fuel_low"

        if report.undercut_threat and not self._undercut_active:
            self._undercut_active = True
            return True, "undercut_threat"
        if not report.undercut_threat and self._undercut_active:
            # Reset so a new threat can fire later.
            self._undercut_active = False

        if report.position is not None and self._last_position is not None and report.position != self._last_position:
            old = self._last_position
            self._last_position = report.position
            return True, f"position_change_from_{old}_to_{report.position}"
        if self._last_position is None and report.position is not None:
            self._last_position = report.position

        if report.overtake_opportunity:
            return True, "overtake_opportunity"

        # Periodic status every PERIODIC_LAP_INTERVAL laps
        if self._last_periodic_lap is None:
            self._last_periodic_lap = report.lap
        elif report.lap - self._last_periodic_lap >= PERIODIC_LAP_INTERVAL:
            self._last_periodic_lap = report.lap
            return True, "periodic_status"

        return False, None

    def generate_message(self, report: SituationReport, trigger: str) -> str:
        report_dict = _round_floats(asdict(report))
        msg = self._llm_call(trigger, json.dumps(report_dict, default=str))
        if not msg or msg.strip().upper() == "NO_COMMENT":
            return ""
        self._last_message_session_t = self._latest_session_t
        return msg.strip()


# --- LLM ----------------------------------------------------------------------

def _default_llm_call(api_key: str | None) -> Callable[[str, str], str]:
    """Returns a callable (trigger, report_json) -> str using Gemini."""

    def _call(trigger: str, report_json: str) -> str:
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY).")
        client = genai.Client(api_key=key)
        user_text = (
            f"Trigger: {trigger}\n\nSituation report:\n```json\n{report_json}\n```\n\nSpeak now."
        )
        resp = client.models.generate_content(
            model=_MODEL,
            contents=user_text,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                max_output_tokens=200,
                # Disable thinking — for one/two-sentence radio calls we don't
                # need the model to deliberate; it eats the output budget.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return (resp.text or "").strip()

    return _call


def _round_floats(obj, ndigits: int = 2):
    """Round all floats inside a nested dict/list. Keeps the LLM's quoted numbers tidy."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, ndigits) for v in obj]
    return obj
