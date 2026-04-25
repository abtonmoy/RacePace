"""SimHub fallback adapter.

SimHub is a Windows app that normalizes telemetry across ~30 sims and
exposes it via shared memory + a configurable WebSocket server. This
adapter consumes the SimHub WebSocket "Game Raw Data" feed (or the
"Custom Properties" plugin output) and maps it onto a `TelemetryFrame`.

Trade-offs (read this before reaching for SimHub):
- ✅ Instant coverage for iRacing, rFactor 2, AMS2, RBR, LMU, etc.
- ❌ Requires SimHub running on Windows.
- ❌ SimHub exposes only what *it* has mapped from each sim — fields are
   often sparse, especially per-tyre data and gap_ahead/behind.
- ❌ Update rate is whatever SimHub offers, typically 60Hz but
   workload-dependent.

Recommend a native adapter where one exists; SimHub is the long tail.

Wire format:
    SimHub sends JSON messages on the WebSocket connection. Each message
    carries a snapshot of subscribed properties keyed by SimHub's
    property names. We map a known subset to TelemetryFrame fields and
    park everything else in `extras`.

Implementation note: this adapter uses the optional `websocket-client`
package (sync). Install via the [simhub] extra:
    uv sync --extra simhub

The SimHub property name list below is conservative; expand as you
encounter more sim/property combinations in the wild.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

from racepace.adapters.base import AbstractAdapter, RateLimiter
from racepace.schema import SessionInfo, TelemetryFrame


# SimHub Property → TelemetryFrame field. (k, scale, offset) — value applied as v*scale + offset.
_PROPERTY_MAP: dict[str, tuple[str, float, float]] = {
    "SpeedKmh":          ("speed_kph", 1.0, 0.0),
    "Throttle":          ("throttle_pct", 1.0, 0.0),  # SimHub gives 0-100 already
    "Brake":             ("brake_pct", 1.0, 0.0),
    "Clutch":            ("clutch_pct", 1.0, 0.0),
    "Gear":              ("gear", 1.0, 0.0),
    "Rpms":              ("rpm", 1.0, 0.0),
    "CurrentLap":        ("lap_number", 1.0, 0.0),
    "TrackPositionPercent": ("lap_distance_pct", 0.01, 0.0),  # SimHub reports 0-100%
    "Fuel":              ("fuel_kg", 1.0, 0.0),  # litres on most sims; convert if known sim==acc
    "Position":          ("position", 1.0, 0.0),
    "LastLapTime":       ("last_lap_time_s", 1.0, 0.0),
    "BestLapTime":       ("best_lap_time_s", 1.0, 0.0),
    "CurrentSectorIndex": ("current_sector", 1.0, 0.0),
}

_TYRE_TEMP_MAP = {
    "TyreTemperatureFrontLeft":  "fl",
    "TyreTemperatureFrontRight": "fr",
    "TyreTemperatureRearLeft":   "rl",
    "TyreTemperatureRearRight":  "rr",
}
_TYRE_PRESS_MAP = {
    "TyrePressureFrontLeft":  "fl",
    "TyrePressureFrontRight": "fr",
    "TyrePressureRearLeft":   "rl",
    "TyrePressureRearRight":  "rr",
}
_TYRE_WEAR_MAP = {
    "TyreWearFrontLeft":  "fl",
    "TyreWearFrontRight": "fr",
    "TyreWearRearLeft":   "rl",
    "TyreWearRearRight":  "rr",
}


def map_simhub_payload(payload: dict[str, Any]) -> tuple[dict, dict]:
    """Returns (telemetry_kwargs, session_kwargs) for the latest SimHub snapshot.

    Pure function; the adapter and tests both call it.
    """
    tel: dict[str, Any] = {"extras": {}}
    sess: dict[str, Any] = {}

    for k, v in payload.items():
        if v is None:
            continue
        if k in _PROPERTY_MAP:
            field, scale, off = _PROPERTY_MAP[k]
            try:
                val = float(v) * scale + off
            except (TypeError, ValueError):
                continue
            if field == "gear" or field == "lap_number" or field == "position" or field == "current_sector":
                tel[field] = int(val)
            else:
                tel[field] = val
        elif k in _TYRE_TEMP_MAP:
            tel.setdefault("tyre_temp_c", {})[_TYRE_TEMP_MAP[k]] = float(v)
        elif k in _TYRE_PRESS_MAP:
            tel.setdefault("tyre_pressure_psi", {})[_TYRE_PRESS_MAP[k]] = float(v)
        elif k in _TYRE_WEAR_MAP:
            tel.setdefault("tyre_wear_pct", {})[_TYRE_WEAR_MAP[k]] = float(v)
        elif k == "TrackName":
            sess["track"] = str(v)
        elif k == "CarModel":
            sess["car"] = str(v)
        elif k == "SessionTypeName":
            v_low = str(v).lower()
            if "race" in v_low: sess["session_type"] = "race"
            elif "qual" in v_low: sess["session_type"] = "qualifying"
            elif "practice" in v_low: sess["session_type"] = "practice"
            else: sess["session_type"] = "hotlap"
        elif k == "Weather":
            sess["weather"] = str(v)
        elif k == "TrackTemperature":
            try: sess["track_temp_c"] = float(v)
            except (TypeError, ValueError): pass
        elif k == "AirTemperature":
            try: sess["air_temp_c"] = float(v)
            except (TypeError, ValueError): pass
        elif k == "TotalLaps":
            try: sess["total_laps"] = int(v)
            except (TypeError, ValueError): pass
        else:
            tel["extras"][f"simhub.{k}"] = v if isinstance(v, (int, float, str, bool)) else str(v)
    return tel, sess


class SimHubAdapter(AbstractAdapter):
    sim_name = "simhub"

    def __init__(
        self,
        url: str = "ws://127.0.0.1:8888/",
        target_hz: float = 30.0,
    ) -> None:
        self.url = url
        self.target_hz = target_hz
        self._ws = None
        self._limiter = RateLimiter(self.target_hz)
        self._t0_monotonic: float | None = None
        self._session_id = str(uuid.uuid4())
        self._latest_payload: dict[str, Any] = {}

    def connect(self) -> None:
        try:
            from websocket import create_connection  # optional dep — `websocket-client`
        except ImportError as e:
            raise RuntimeError(
                "SimHub adapter needs the websocket-client package. "
                "Install via `uv sync --extra simhub`."
            ) from e
        self._ws = create_connection(self.url, timeout=2.0)
        self._t0_monotonic = time.monotonic()

    def disconnect(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def read_session_info(self) -> SessionInfo:
        # Pull a few payloads to learn track/car/session.
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            self._read_one()
            if "track" in self._latest_session_kwargs():
                break
        sess = self._latest_session_kwargs()
        return SessionInfo(
            session_id=self._session_id,
            sim=self.sim_name,
            started_at=datetime.now(timezone.utc),
            track=sess.get("track"),
            car=sess.get("car"),
            session_type=sess.get("session_type"),
            weather=sess.get("weather"),
            track_temp_c=sess.get("track_temp_c"),
            air_temp_c=sess.get("air_temp_c"),
            total_laps=sess.get("total_laps"),
        )

    def stream_frames(self) -> Iterator[TelemetryFrame]:
        if self._ws is None:
            raise RuntimeError("connect() before stream_frames()")
        if self._t0_monotonic is None:
            self._t0_monotonic = time.monotonic()
        while True:
            self._read_one()
            if not self._limiter.should_emit():
                continue
            frame = self._build_frame()
            if frame is not None:
                yield frame

    def _read_one(self) -> None:
        if self._ws is None:
            return
        try:
            raw = self._ws.recv()
        except Exception:
            return
        if not raw:
            return
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return
        if isinstance(payload, dict):
            self._latest_payload = payload

    def _latest_session_kwargs(self) -> dict[str, Any]:
        _, sess = map_simhub_payload(self._latest_payload)
        return sess

    def _build_frame(self) -> TelemetryFrame | None:
        if not self._latest_payload:
            return None
        ts = time.monotonic() - (self._t0_monotonic or time.monotonic())
        tel_kwargs, _ = map_simhub_payload(self._latest_payload)
        try:
            return TelemetryFrame(
                timestamp_s=ts,
                lap_number=tel_kwargs.get("lap_number") or 1,
                **{k: v for k, v in tel_kwargs.items() if k not in ("lap_number",)},
            )
        except Exception:
            return None
