"""SimHub property mapping (pure function — no SimHub install required)."""

from __future__ import annotations

from racepace.adapters.simhub import map_simhub_payload


def test_basic_telemetry_fields_map_through():
    payload = {
        "SpeedKmh": 200.0,
        "Throttle": 88.0,
        "Brake": 0.0,
        "Gear": 5,
        "Rpms": 7800,
        "CurrentLap": 3,
        "Position": 4,
    }
    tel, sess = map_simhub_payload(payload)
    assert tel["speed_kph"] == 200.0
    assert tel["throttle_pct"] == 88.0
    assert tel["gear"] == 5
    assert tel["lap_number"] == 3
    assert tel["position"] == 4
    assert sess == {}


def test_tyre_dicts_assembled():
    payload = {
        "TyreTemperatureFrontLeft": 80.0,
        "TyreTemperatureFrontRight": 82.0,
        "TyrePressureFrontLeft": 27.5,
    }
    tel, _ = map_simhub_payload(payload)
    assert tel["tyre_temp_c"]["fl"] == 80.0
    assert tel["tyre_temp_c"]["fr"] == 82.0
    assert tel["tyre_pressure_psi"]["fl"] == 27.5
    assert "rl" not in tel["tyre_temp_c"]   # only what was provided


def test_session_fields_recognized():
    payload = {
        "TrackName": "Spa",
        "CarModel": "Ferrari 296 GT3",
        "SessionTypeName": "Race",
        "Weather": "Cloudy",
        "TrackTemperature": 28.5,
        "TotalLaps": 42,
    }
    _, sess = map_simhub_payload(payload)
    assert sess["track"] == "Spa"
    assert sess["car"] == "Ferrari 296 GT3"
    assert sess["session_type"] == "race"
    assert sess["weather"] == "Cloudy"
    assert sess["track_temp_c"] == 28.5
    assert sess["total_laps"] == 42


def test_unknown_property_falls_into_extras():
    payload = {"WeirdSimSpecificThing": 42, "SpeedKmh": 100.0}
    tel, _ = map_simhub_payload(payload)
    assert tel["speed_kph"] == 100.0
    assert tel["extras"]["simhub.WeirdSimSpecificThing"] == 42


def test_lap_distance_pct_is_normalized_from_simhub_percentage():
    payload = {"TrackPositionPercent": 50.0}  # SimHub gives 0-100
    tel, _ = map_simhub_payload(payload)
    assert tel["lap_distance_pct"] == 0.5
