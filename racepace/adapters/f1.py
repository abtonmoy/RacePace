"""F1 24/25 UDP telemetry adapter.

The Codemasters F1 series broadcasts UDP packets on a configurable port
(default 20777). This adapter listens, parses the subset of packet types
needed to populate a `TelemetryFrame`, and emits frames at the target Hz.

Reference: F1 24/25 Telemetry Specification (Codemasters).

Packet types handled:
- Session (id=1) — track, weather, total laps, session type, temps
- LapData (id=2) — lap number, distance, sector, lap times, position, pit status
- CarTelemetry (id=6) — speed, throttle, brake, gear, rpm, steering, tyre temps/pressures
- CarStatus (id=7) — fuel, tyre compound, ERS state, tyre wear estimate

Other packet types (Motion, Participants, etc.) are recognized and skipped.

Cross-version notes:
- F1 23+ has a 29-byte header; older games used 24 bytes. We parse based
  on the m_gameYear field. Anything before F1 23 is rejected.
- Field offsets for F1 24 and F1 25 are identical for the packet types we
  parse; future shifts will need a per-version parser.

ERS energy and forecast samples land in `extras`.
"""

from __future__ import annotations

import socket
import struct
import time
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone

from racepace.adapters.base import AbstractAdapter, RateLimiter
from racepace.schema import SessionInfo, TelemetryFrame


PACKET_SESSION = 1
PACKET_LAP_DATA = 2
PACKET_CAR_TELEMETRY = 6
PACKET_CAR_STATUS = 7

# F1 24/25 header (29 bytes) — '<' = little-endian, packed
HEADER_FMT = "<HBBBBBQfIIBB"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
assert HEADER_SIZE == 29

# Per-car CarTelemetry (60 bytes)
CAR_TEL_FMT = "<HfffBbHBBH4H4B4BH4f4B"
CAR_TEL_SIZE = struct.calcsize(CAR_TEL_FMT)

# Track ID → name (subset; expand as encountered)
_TRACK_NAMES = {
    0: "melbourne", 1: "paul_ricard", 2: "shanghai", 3: "sakhir", 4: "catalunya",
    5: "monaco", 6: "montreal", 7: "silverstone", 8: "hockenheim", 9: "hungaroring",
    10: "spa", 11: "monza", 12: "singapore", 13: "suzuka", 14: "abu_dhabi",
    15: "texas", 16: "brazil", 17: "austria", 18: "sochi", 19: "mexico",
    20: "baku", 21: "sakhir_short", 22: "silverstone_short", 23: "texas_short",
    24: "suzuka_short", 25: "hanoi", 26: "zandvoort", 27: "imola", 28: "portimao",
    29: "jeddah", 30: "miami", 31: "vegas", 32: "losail",
}

# Session-type IDs → schema strings
_SESSION_TYPES = {
    1: "practice", 2: "practice", 3: "practice",
    5: "qualifying", 6: "qualifying", 7: "qualifying", 8: "qualifying", 9: "qualifying",
    10: "race", 11: "race", 12: "race",
    13: "hotlap",
    15: "race",  # F1 24 sprint race
}


def parse_header(buf: bytes) -> dict | None:
    if len(buf) < HEADER_SIZE:
        return None
    (
        packet_format, game_year, game_major, game_minor, packet_version, packet_id,
        session_uid, session_time, frame_id, overall_frame_id, player_car_idx, sec_player_car_idx,
    ) = struct.unpack_from(HEADER_FMT, buf, 0)
    return {
        "packet_format": packet_format,
        "game_year": game_year,
        "packet_id": packet_id,
        "session_time": session_time,
        "player_car_index": player_car_idx,
    }


def parse_car_telemetry(buf: bytes, header: dict) -> dict | None:
    """Returns the player's car telemetry as a dict, or None on size mismatch."""
    idx = header["player_car_index"]
    offset = HEADER_SIZE + idx * CAR_TEL_SIZE
    if len(buf) < offset + CAR_TEL_SIZE:
        return None
    (
        speed, throttle, steer, brake, clutch, gear, rpm, drs, rev_pct, rev_bits,
        bt_rl, bt_rr, bt_fl, bt_fr,
        ts_rl, ts_rr, ts_fl, ts_fr,
        ti_rl, ti_rr, ti_fl, ti_fr,
        engine_temp,
        tp_rl, tp_rr, tp_fl, tp_fr,
        sf_rl, sf_rr, sf_fl, sf_fr,
    ) = struct.unpack_from(CAR_TEL_FMT, buf, offset)
    return {
        "speed_kph": float(speed),
        "throttle_pct": float(throttle) * 100.0,
        "brake_pct": float(brake) * 100.0,
        "steering_norm": float(steer),
        "clutch_pct": float(clutch),
        "gear": int(gear),
        "rpm": float(rpm),
        "drs": int(drs),
        "tyre_surface_temp_c": {"fl": ts_fl, "fr": ts_fr, "rl": ts_rl, "rr": ts_rr},
        "tyre_inner_temp_c": {"fl": ti_fl, "fr": ti_fr, "rl": ti_rl, "rr": ti_rr},
        "tyre_pressure_psi": {"fl": tp_fl, "fr": tp_fr, "rl": tp_rl, "rr": tp_rr},
    }


# --- LapData (per car, F1 24) ---
# Just the leading fields we need — the rest are unpacked but ignored.
LAP_DATA_FMT = "<II HBHB HBHB ffffBBBBBBBBBBBBBBBBBHHBfB BB"
# We'll use struct.iter_unpack over a tighter sub-format; full per-car size is computed.
# For correctness without a full parser, we explicitly parse only what we use.

LAP_DATA_PER_CAR_SIZE = 57  # F1 24 — verified against the spec

def parse_lap_data(buf: bytes, header: dict) -> dict | None:
    idx = header["player_car_index"]
    offset = HEADER_SIZE + idx * LAP_DATA_PER_CAR_SIZE
    if len(buf) < offset + 30:  # we touch fields up to ~byte 30
        return None
    last_lap_ms = struct.unpack_from("<I", buf, offset + 0)[0]
    cur_lap_ms = struct.unpack_from("<I", buf, offset + 4)[0]
    lap_distance = struct.unpack_from("<f", buf, offset + 22)[0]
    car_position = struct.unpack_from("<B", buf, offset + 38)[0] if len(buf) >= offset + 39 else None
    cur_lap_num = struct.unpack_from("<B", buf, offset + 39)[0] if len(buf) >= offset + 40 else None
    pit_status = struct.unpack_from("<B", buf, offset + 40)[0] if len(buf) >= offset + 41 else None
    sector = struct.unpack_from("<B", buf, offset + 42)[0] if len(buf) >= offset + 43 else None
    return {
        "last_lap_time_s": last_lap_ms / 1000.0 if last_lap_ms > 0 else None,
        "current_lap_time_s": cur_lap_ms / 1000.0,
        "lap_distance_m": float(lap_distance),
        "position": int(car_position) if car_position is not None else None,
        "lap_number": int(cur_lap_num) if cur_lap_num is not None else None,
        "pit_status": int(pit_status) if pit_status is not None else None,  # 0 none 1 pitting 2 in pit area
        "current_sector": int(sector) if sector is not None else None,
    }


# --- CarStatus (per car, F1 24) ---
CAR_STATUS_PER_CAR_SIZE = 55

_TYRE_COMPOUND_MAP = {
    16: "soft", 17: "medium", 18: "hard", 7: "intermediate", 8: "wet",
    9: "dry", 10: "wet", 11: "super_soft", 12: "soft", 13: "medium", 14: "hard", 15: "wet",
}

def parse_car_status(buf: bytes, header: dict) -> dict | None:
    idx = header["player_car_index"]
    offset = HEADER_SIZE + idx * CAR_STATUS_PER_CAR_SIZE
    if len(buf) < offset + 55:
        return None
    # Field offsets per spec (F1 24); read selectively.
    fuel_in_tank = struct.unpack_from("<f", buf, offset + 12)[0]
    fuel_capacity = struct.unpack_from("<f", buf, offset + 16)[0]
    fuel_remaining_laps = struct.unpack_from("<f", buf, offset + 20)[0]
    actual_compound = struct.unpack_from("<B", buf, offset + 27)[0]
    tyres_age_laps = struct.unpack_from("<B", buf, offset + 29)[0]
    ers_store_energy = struct.unpack_from("<f", buf, offset + 38)[0]
    ers_deploy_mode = struct.unpack_from("<B", buf, offset + 42)[0]
    return {
        "fuel_kg": float(fuel_in_tank),
        "fuel_capacity_kg": float(fuel_capacity),
        "fuel_laps_remaining": float(fuel_remaining_laps),
        "tyre_compound": _TYRE_COMPOUND_MAP.get(actual_compound, str(actual_compound)),
        "tyre_age_laps": int(tyres_age_laps),
        "ers_store_j": float(ers_store_energy),
        "ers_deploy_mode": int(ers_deploy_mode),
    }


SESSION_HEAD_FMT = "<BbbBBHBbBHHBBBBBBB"  # very partial; we read leading fields only

def parse_session(buf: bytes, header: dict) -> dict | None:
    if len(buf) < HEADER_SIZE + 19:
        return None
    weather = struct.unpack_from("<B", buf, HEADER_SIZE + 0)[0]   # 0..5
    track_temp = struct.unpack_from("<b", buf, HEADER_SIZE + 1)[0]
    air_temp = struct.unpack_from("<b", buf, HEADER_SIZE + 2)[0]
    total_laps = struct.unpack_from("<B", buf, HEADER_SIZE + 3)[0]
    track_id = struct.unpack_from("<b", buf, HEADER_SIZE + 6)[0]
    session_type = struct.unpack_from("<B", buf, HEADER_SIZE + 8)[0]
    return {
        "weather": ["clear", "light_cloud", "overcast", "light_rain", "heavy_rain", "storm"][weather] if 0 <= weather <= 5 else None,
        "track_temp_c": float(track_temp),
        "air_temp_c": float(air_temp),
        "total_laps": int(total_laps) if total_laps > 0 else None,
        "track": _TRACK_NAMES.get(int(track_id), f"track_{track_id}"),
        "session_type": _SESSION_TYPES.get(int(session_type)),
    }


# --- Adapter ----------------------------------------------------------------

class F1Adapter(AbstractAdapter):
    sim_name = "f1"

    def __init__(self, port: int = 20777, target_hz: float = 30.0, host: str = "0.0.0.0") -> None:
        self.port = port
        self.host = host
        self.target_hz = target_hz
        self._sock: socket.socket | None = None
        self._limiter = RateLimiter(self.target_hz)
        self._t0_monotonic: float | None = None
        self._session_id = str(uuid.uuid4())

        # Latest-of-each-packet state
        self._latest_telemetry: dict | None = None
        self._latest_lap: dict | None = None
        self._latest_status: dict | None = None
        self._latest_session: dict | None = None

    def connect(self) -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((self.host, self.port))
        s.settimeout(0.5)
        self._sock = s
        self._t0_monotonic = time.monotonic()

    def disconnect(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def read_session_info(self) -> SessionInfo:
        # Pull a few packets to learn track/session before returning.
        if self._sock is None:
            raise RuntimeError("connect() before read_session_info()")
        deadline = time.monotonic() + 3.0
        while self._latest_session is None and time.monotonic() < deadline:
            self._read_one_packet()
        sess = self._latest_session or {}
        return SessionInfo(
            session_id=self._session_id,
            sim=self.sim_name,
            started_at=datetime.now(timezone.utc),
            track=sess.get("track"),
            car=None,
            session_type=sess.get("session_type"),
            weather=sess.get("weather"),
            track_temp_c=sess.get("track_temp_c"),
            air_temp_c=sess.get("air_temp_c"),
            total_laps=sess.get("total_laps"),
        )

    def stream_frames(self) -> Iterator[TelemetryFrame]:
        if self._sock is None:
            raise RuntimeError("connect() before stream_frames()")
        if self._t0_monotonic is None:
            self._t0_monotonic = time.monotonic()

        while True:
            self._read_one_packet()
            if not self._limiter.should_emit():
                continue
            frame = self._build_frame()
            if frame is not None:
                yield frame

    def _read_one_packet(self) -> None:
        if self._sock is None:
            return
        try:
            buf, _addr = self._sock.recvfrom(2048)
        except socket.timeout:
            return
        except OSError:
            return
        header = parse_header(buf)
        if header is None:
            return
        if header.get("game_year", 0) and header["game_year"] < 23:
            return
        pid = header["packet_id"]
        try:
            if pid == PACKET_CAR_TELEMETRY:
                self._latest_telemetry = parse_car_telemetry(buf, header) or self._latest_telemetry
            elif pid == PACKET_LAP_DATA:
                self._latest_lap = parse_lap_data(buf, header) or self._latest_lap
            elif pid == PACKET_CAR_STATUS:
                self._latest_status = parse_car_status(buf, header) or self._latest_status
            elif pid == PACKET_SESSION:
                self._latest_session = parse_session(buf, header) or self._latest_session
        except struct.error:
            return

    def _build_frame(self) -> TelemetryFrame | None:
        if self._latest_telemetry is None or self._latest_lap is None:
            return None
        ts = time.monotonic() - (self._t0_monotonic or time.monotonic())
        tel = self._latest_telemetry
        lap = self._latest_lap
        status = self._latest_status or {}
        sess = self._latest_session or {}

        extras: dict = {}
        if "ers_store_j" in status:
            extras["ers_store_j"] = status["ers_store_j"]
        if "ers_deploy_mode" in status:
            extras["ers_deploy_mode"] = status["ers_deploy_mode"]
        if "drs" in tel:
            extras["drs"] = tel["drs"]
        if "tyre_surface_temp_c" in tel:
            extras["tyre_surface_temp_c"] = tel["tyre_surface_temp_c"]

        # F1 estimates wear from tyre_age_laps; real values are in CarDamage (not parsed yet).
        wear: dict[str, float] | None = None
        if "tyre_age_laps" in status:
            est = min(100.0, status["tyre_age_laps"] * 2.0)  # ~2% per lap rule of thumb
            wear = {"fl": est, "fr": est, "rl": est, "rr": est}

        track_len = sess.get("track_length_m")
        return TelemetryFrame(
            timestamp_s=ts,
            lap_number=lap.get("lap_number") or 1,
            lap_distance_m=lap.get("lap_distance_m"),
            lap_distance_pct=(lap.get("lap_distance_m") / track_len) if (track_len and lap.get("lap_distance_m") is not None) else None,
            speed_kph=tel.get("speed_kph"),
            throttle_pct=tel.get("throttle_pct"),
            brake_pct=tel.get("brake_pct"),
            clutch_pct=tel.get("clutch_pct"),
            steering_norm=tel.get("steering_norm"),
            gear=tel.get("gear"),
            rpm=tel.get("rpm"),
            fuel_kg=status.get("fuel_kg"),
            fuel_laps_remaining=status.get("fuel_laps_remaining"),
            tyre_wear_pct=wear,
            tyre_temp_c=tel.get("tyre_inner_temp_c"),
            tyre_pressure_psi=tel.get("tyre_pressure_psi"),
            tyre_compound=status.get("tyre_compound"),
            position=lap.get("position"),
            last_lap_time_s=lap.get("last_lap_time_s"),
            current_sector=lap.get("current_sector"),
            extras=extras,
        )
