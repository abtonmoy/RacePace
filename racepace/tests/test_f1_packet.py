"""F1 24/25 UDP packet parser tests using synthesized binary packets."""

from __future__ import annotations

import struct

from racepace.adapters.f1 import (
    CAR_TEL_FMT,
    CAR_TEL_SIZE,
    HEADER_FMT,
    HEADER_SIZE,
    PACKET_CAR_TELEMETRY,
    PACKET_LAP_DATA,
    parse_car_telemetry,
    parse_header,
    parse_lap_data,
)


def _make_header(packet_id: int, player_idx: int = 0) -> bytes:
    return struct.pack(
        HEADER_FMT,
        2024,  # packet_format
        24,    # game_year
        1, 0,  # major.minor
        1,     # packet_version
        packet_id,
        12345678901234,  # session_uid
        12.5,            # session_time
        100,             # frame_id
        100,             # overall_frame_id
        player_idx,
        255,             # secondary
    )


def _make_car_tel_entry(speed=180, throttle=0.85, brake=0.0, steer=0.1, gear=4, rpm=7500):
    """One CarTelemetryData record (60 bytes)."""
    return struct.pack(
        CAR_TEL_FMT,
        speed,                 # H — speed kph
        throttle, steer, brake,  # f f f
        50,                    # B — clutch
        gear,                  # b — gear
        rpm,                   # H — engine_rpm
        0,                     # B — drs
        50,                    # B — rev_lights_pct
        0xFFFF,                # H — rev_lights_bits
        300, 305, 310, 312,    # 4H brakes_temp
        90, 92, 95, 96,        # 4B surface_temp
        85, 87, 90, 91,        # 4B inner_temp
        110,                   # H — engine_temp
        27.5, 27.6, 27.0, 27.1,  # 4f tyre_pressure
        0, 0, 0, 0,            # 4B surface_type
    )


def test_header_parses_round_trip():
    buf = _make_header(PACKET_CAR_TELEMETRY)
    h = parse_header(buf)
    assert h is not None
    assert h["packet_id"] == PACKET_CAR_TELEMETRY
    assert h["player_car_index"] == 0


def test_car_telemetry_parses_player_car():
    header = _make_header(PACKET_CAR_TELEMETRY, player_idx=2)
    cars = b"".join(_make_car_tel_entry(speed=100 + i) for i in range(22))
    trailing = struct.pack("<BBb", 0, 0, 0)
    buf = header + cars + trailing
    parsed_h = parse_header(buf)
    tel = parse_car_telemetry(buf, parsed_h)
    assert tel is not None
    assert tel["speed_kph"] == 102.0  # car index 2 → speed 100 + 2
    assert abs(tel["throttle_pct"] - 85.0) < 1e-3
    assert tel["gear"] == 4
    assert tel["rpm"] == 7500.0
    assert abs(tel["tyre_pressure_psi"]["fl"] - 27.0) < 1e-3
    assert abs(tel["tyre_pressure_psi"]["fr"] - 27.1) < 1e-3


def test_short_buffer_returns_none():
    header = _make_header(PACKET_CAR_TELEMETRY, player_idx=0)
    short = header + b"\x00" * 10
    parsed_h = parse_header(short)
    assert parse_car_telemetry(short, parsed_h) is None


def test_old_game_year_rejected_by_adapter_pre_filter():
    """parse_header doesn't gate on game_year — the adapter does. Sanity-check the field surfaces."""
    buf = struct.pack(
        HEADER_FMT,
        2022, 22, 1, 0, 1, 6, 1, 0.0, 0, 0, 0, 255,
    )
    h = parse_header(buf)
    assert h is not None
    assert h["game_year"] == 22
