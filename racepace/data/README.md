# Real F1 data importers

Two paths to real F1 telemetry, both writing into the same `TelemetryFrame` schema. Both are *importers*, not adapters — they pull a finished session from an HTTP API and write it to a SQLite session DB. From there, all three agents work on it the same way they would on a recorded ACC session.

| | `fastf1_import.py` | `openf1_import.py` |
|---|---|---|
| Source | F1 official + Ergast servers | api.openf1.org (community-run) |
| Library | `fastf1` (PyPI) | stdlib `urllib` |
| Setup | `uv sync --extra fastf1` | None — stdlib only |
| Cache | Local pickle cache (`.fastf1_cache/`) | None (each query hits the API) |
| Sample rate | ~3–4 Hz car + position | ~4 Hz car_data + ~4 Hz location |
| Distance | native field | derived from X/Y deltas (decimeters → m) |
| Brake | boolean (False/True) | int 0 or 100 |
| Gap to ahead | not in `get_telemetry()` | `/v1/intervals` per-frame |
| Position trace | per-lap from `lap.Position` | `/v1/position` (changes only — needs at-or-before lookup) |
| Tyre compound | `lap.Compound` | from `/v1/stints` (importer doesn't fetch yet) |
| Coordinate units | metres | **decimetres** (1/10 m) — divide by 10 in distance integration |

Both importers compute a steering proxy from the X/Y trajectory's heading delta — F1 broadcast does not expose steering wheel position. The L/R sign sometimes flips depending on how the sim's coordinate system orients the track; corner positions are always correct, only the directional label can be wrong. Hand-edit the track map JSON to fix.

## What F1 broadcast does NOT give either importer

- Fuel mass / laps remaining → `frame.fuel_kg = None`
- Tyre wear per corner → `frame.tyre_wear_pct = None` (FastF1 importer puts `tyre_age_laps` in `extras`)
- Tyre temps / pressures → all None
- Steering wheel position → derived as a heading proxy
- Brake pressure → boolean → 0 or 100

The agents handle `None` gracefully (this is the Phase 1 contract). The engineer's fuel triggers stay silent; the analyst says "not reported" for missing fields.

## CLI usage

```bash
# FastF1
uv run racepace fastf1-import --year 2023 --gp Monza --session-type R \
  --driver VER --output ver_monza.db --lap-lo 1 --lap-hi 15

# OpenF1
uv run racepace openf1-import --year 2023 --country Monza --session-name Race \
  --driver-number 1 --output ver_monza_openf1.db --lap-lo 1 --lap-hi 15
```

Driver identification differs:
- FastF1: 3-letter code (`VER`, `HAM`, `LEC`)
- OpenF1: numeric driver number (`1`, `44`, `16`)

Session naming differs:
- FastF1 `--session-type`: `R`, `Q`, `FP1`, `FP2`, `FP3`, `Sprint`
- OpenF1 `--session-name`: `Race`, `Qualifying`, `Practice 1`, `Sprint Qualifying`, etc.

Country/track lookup in OpenF1: prefers exact `location` / `circuit_short_name` match, then substring, then country. Skips cancelled sessions automatically — important because some country names map to multiple circuits (e.g. Italy → Imola + Monza).

## End-to-end on real F1 data

```bash
# 1. Import
racepace openf1-import --year 2023 --country Monza --session-name Race \
  --driver-number 1 --output ver.db --lap-lo 1 --lap-hi 15

# 2. Post-session debrief (Phase 1)
racepace analyze ver.db

# 3. Build a track map and a reference lap
racepace extract-track ver.db --lap 3 --output racepace/tracks/f1/monza.json --track monza
racepace save-reference ver.db --lap 3 --track monza --car "Red Bull Racing"

# 4. Engineer + coach drivers (see racepace/tests/ for the synchronous test driver pattern)
```

## When to use which

- **FastF1** if you want a Python-native API, lots of derived fields, local caching, or are building things on top of a finished session structure.
- **OpenF1** if you want zero install footprint, finer-grained position-change data (catches mid-lap overtakes), or need to integrate from a non-Python service.

Both importers were verified end-to-end against Verstappen's 2023 Italian GP race laps 1–15. See the project README for the test results.
