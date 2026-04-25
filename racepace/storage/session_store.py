"""SQLite-backed session storage.

One file per recording. Two tables: `sessions` (one row per session,
columns mirror SessionInfo) and `frames` (one row per TelemetryFrame
with a session_id FK; dict fields stored as JSON text).

A `meta` table records the schema_version the file was written under.
SessionReader refuses to load files written under an incompatible
version (different MAJOR component of SCHEMA_VERSION).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from racepace.schema import SCHEMA_VERSION, SessionInfo, TelemetryFrame


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    sim TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    track TEXT,
    car TEXT,
    session_type TEXT,
    weather TEXT,
    track_temp_c REAL,
    air_temp_c REAL,
    total_laps INTEGER
);

CREATE TABLE IF NOT EXISTS frames (
    session_id TEXT NOT NULL,
    timestamp_s REAL NOT NULL,
    lap_number INTEGER NOT NULL,
    lap_distance_m REAL,
    lap_distance_pct REAL,
    speed_kph REAL,
    throttle_pct REAL,
    brake_pct REAL,
    clutch_pct REAL,
    steering_norm REAL,
    gear INTEGER,
    rpm REAL,
    fuel_kg REAL,
    fuel_laps_remaining REAL,
    tyre_wear_pct TEXT,
    tyre_temp_c TEXT,
    tyre_pressure_psi TEXT,
    tyre_compound TEXT,
    position INTEGER,
    gap_ahead_s REAL,
    gap_behind_s REAL,
    last_lap_time_s REAL,
    best_lap_time_s REAL,
    current_sector INTEGER,
    extras TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS frames_session_ts ON frames(session_id, timestamp_s);
CREATE INDEX IF NOT EXISTS frames_session_lap ON frames(session_id, lap_number);
"""


class SchemaVersionMismatch(RuntimeError):
    """Raised when a session file was written under an incompatible schema."""


def _major(v: str) -> str:
    return v.split(".", 1)[0]


def _dumps(d: dict[str, Any] | None) -> str | None:
    return json.dumps(d) if d is not None else None


def _loads(s: str | None) -> dict[str, Any] | None:
    return json.loads(s) if s else None


class SessionWriter:
    """Append-only writer for one session file.

    Caller responsibility: call write_session() once before write_frame().
    Frames are buffered into transactions of `flush_every` for throughput.
    """

    def __init__(self, path: str | Path, flush_every: int = 200) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
            (SCHEMA_VERSION,),
        )
        self._conn.commit()
        self._flush_every = flush_every
        self._pending = 0
        self._session_id: str | None = None

    def write_session(self, info: SessionInfo) -> None:
        self._session_id = info.session_id
        self._conn.execute(
            """INSERT OR REPLACE INTO sessions
               (session_id, sim, started_at, ended_at, track, car, session_type,
                weather, track_temp_c, air_temp_c, total_laps)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                info.session_id,
                info.sim,
                info.started_at.isoformat(),
                info.ended_at.isoformat() if info.ended_at else None,
                info.track,
                info.car,
                info.session_type,
                info.weather,
                info.track_temp_c,
                info.air_temp_c,
                info.total_laps,
            ),
        )
        self._conn.commit()

    def update_session_end(self, ended_at: datetime, total_laps: int | None = None) -> None:
        if self._session_id is None:
            raise RuntimeError("write_session() must be called before update_session_end()")
        self._conn.execute(
            "UPDATE sessions SET ended_at = ?, total_laps = COALESCE(?, total_laps) WHERE session_id = ?",
            (ended_at.isoformat(), total_laps, self._session_id),
        )
        self._conn.commit()

    def write_frame(self, frame: TelemetryFrame) -> None:
        if self._session_id is None:
            raise RuntimeError("write_session() must be called before write_frame()")
        self._conn.execute(
            """INSERT INTO frames
               (session_id, timestamp_s, lap_number, lap_distance_m, lap_distance_pct,
                speed_kph, throttle_pct, brake_pct, clutch_pct, steering_norm, gear, rpm,
                fuel_kg, fuel_laps_remaining,
                tyre_wear_pct, tyre_temp_c, tyre_pressure_psi, tyre_compound,
                position, gap_ahead_s, gap_behind_s,
                last_lap_time_s, best_lap_time_s, current_sector, extras)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self._session_id,
                frame.timestamp_s,
                frame.lap_number,
                frame.lap_distance_m,
                frame.lap_distance_pct,
                frame.speed_kph,
                frame.throttle_pct,
                frame.brake_pct,
                frame.clutch_pct,
                frame.steering_norm,
                frame.gear,
                frame.rpm,
                frame.fuel_kg,
                frame.fuel_laps_remaining,
                _dumps(frame.tyre_wear_pct),
                _dumps(frame.tyre_temp_c),
                _dumps(frame.tyre_pressure_psi),
                frame.tyre_compound,
                frame.position,
                frame.gap_ahead_s,
                frame.gap_behind_s,
                frame.last_lap_time_s,
                frame.best_lap_time_s,
                frame.current_sector,
                json.dumps(frame.extras or {}),
            ),
        )
        self._pending += 1
        if self._pending >= self._flush_every:
            self._conn.commit()
            self._pending = 0

    def close(self) -> None:
        if self._conn is None:
            return
        try:
            self._conn.commit()
        finally:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]

    def __enter__(self) -> SessionWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class SessionReader:
    """Read-only reader for a session file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        self._conn.row_factory = sqlite3.Row
        self._check_version()

    def _check_version(self) -> None:
        cur = self._conn.execute("SELECT value FROM meta WHERE key = 'schema_version'")
        row = cur.fetchone()
        if row is None:
            raise SchemaVersionMismatch(f"{self.path}: no schema_version recorded.")
        file_v = row["value"]
        if _major(file_v) != _major(SCHEMA_VERSION):
            raise SchemaVersionMismatch(
                f"{self.path}: written under schema {file_v}, current is {SCHEMA_VERSION}."
            )

    def list_sessions(self) -> list[SessionInfo]:
        cur = self._conn.execute("SELECT * FROM sessions ORDER BY started_at")
        return [_row_to_session(r) for r in cur.fetchall()]

    def latest_session_id(self) -> str | None:
        cur = self._conn.execute(
            "SELECT session_id FROM sessions ORDER BY started_at DESC LIMIT 1"
        )
        row = cur.fetchone()
        return row["session_id"] if row else None

    def load_session(self, session_id: str) -> tuple[SessionInfo, list[TelemetryFrame]]:
        s_cur = self._conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        )
        s_row = s_cur.fetchone()
        if s_row is None:
            raise KeyError(session_id)
        info = _row_to_session(s_row)

        f_cur = self._conn.execute(
            "SELECT * FROM frames WHERE session_id = ? ORDER BY timestamp_s",
            (session_id,),
        )
        frames = [_row_to_frame(r) for r in f_cur.fetchall()]
        return info, frames

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SessionReader:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _row_to_session(r: sqlite3.Row) -> SessionInfo:
    return SessionInfo(
        session_id=r["session_id"],
        sim=r["sim"],
        started_at=datetime.fromisoformat(r["started_at"]),
        ended_at=datetime.fromisoformat(r["ended_at"]) if r["ended_at"] else None,
        track=r["track"],
        car=r["car"],
        session_type=r["session_type"],
        weather=r["weather"],
        track_temp_c=r["track_temp_c"],
        air_temp_c=r["air_temp_c"],
        total_laps=r["total_laps"],
    )


def _row_to_frame(r: sqlite3.Row) -> TelemetryFrame:
    return TelemetryFrame(
        timestamp_s=r["timestamp_s"],
        lap_number=r["lap_number"],
        lap_distance_m=r["lap_distance_m"],
        lap_distance_pct=r["lap_distance_pct"],
        speed_kph=r["speed_kph"],
        throttle_pct=r["throttle_pct"],
        brake_pct=r["brake_pct"],
        clutch_pct=r["clutch_pct"],
        steering_norm=r["steering_norm"],
        gear=r["gear"],
        rpm=r["rpm"],
        fuel_kg=r["fuel_kg"],
        fuel_laps_remaining=r["fuel_laps_remaining"],
        tyre_wear_pct=_loads(r["tyre_wear_pct"]),
        tyre_temp_c=_loads(r["tyre_temp_c"]),
        tyre_pressure_psi=_loads(r["tyre_pressure_psi"]),
        tyre_compound=r["tyre_compound"],
        position=r["position"],
        gap_ahead_s=r["gap_ahead_s"],
        gap_behind_s=r["gap_behind_s"],
        last_lap_time_s=r["last_lap_time_s"],
        best_lap_time_s=r["best_lap_time_s"],
        current_sector=r["current_sector"],
        extras=json.loads(r["extras"]) if r["extras"] else {},
    )
