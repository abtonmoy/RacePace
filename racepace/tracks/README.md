# Track maps

A track map is the coach's reference layout for one circuit + sim. JSON, hand-editable.

```
tracks/
  acc/
    spa.json
    monza.json
    synthetic_test.json   # committed test fixture
  f1/
    monza.json
    synthetic_test.json
```

## Schema

```jsonc
{
  "track": "monza",
  "sim": "f1",
  "car": "Red Bull Racing",
  "total_length_m": 5719.0,
  "corners": [
    {
      "id": 1,
      "name": null,                 // hand-edit: e.g. "T1 - Rettifilo"
      "direction": "R",             // "L" or "R"
      "severity": 5,                // 1 (flat-out near-corner) to 6 (heavy braking, low gear)
      "brake_point_m": 654.0,       // distance from SF
      "apex_m": 918.0,
      "exit_m": 1000.0,
      "target_min_speed_kph": 68.1, // observed at apex on the reference lap
      "target_gear": 2,
      "flat": false,                // true if the reference brake was always 0
      "notes": null                 // hand-edit: e.g. "don't cut the second kerb"
    }
  ]
}
```

## Generating one

```bash
# 1. Record a clean reference lap (or import one from FastF1/OpenF1)
racepace record --sim acc --output spa.db
# or:  racepace openf1-import --year 2023 --country Belgium --session-name Race --driver-number 1 \
#                              --output spa.db --lap-lo 1 --lap-hi 5

# 2. Extract the corners from one specific lap (use a clean fast lap)
racepace extract-track spa.db --lap 4 --output racepace/tracks/acc/spa.json --track spa
```

## The hand-edit step

Auto-extraction gets the geometry right (brake point, apex, exit, target speed) but cannot name corners. Edit the JSON to add:

- `name` — corner names ("Eau Rouge", "Pouhon", "Curva Grande", "T1 — Rettifilo"). The slow-loop coach uses these in its prompts.
- `notes` — driver-facing reminders ("don't cut the kerb", "throttle through, no lift", "trail brake long")

This is a deliberate human-in-the-loop step. Automatic naming from public data is unreliable and the edit takes minutes per track.

## Direction can flip on real F1 data

The FastF1 / OpenF1 importers derive the steering proxy from the X/Y trajectory's heading delta. Depending on the sim's coordinate convention, the L/R sign sometimes inverts. Apex positions are always correct; just check the direction field after extraction and flip if needed.

## Severity scale

| Severity | Min speed | Brake load | Examples |
|---|---|---|---|
| 1 | ≥220 kph, no brake | none | Eau Rouge, Curva Grande (flat-out kinks) |
| 2 | ≥180 kph | <30% | High-speed sweepers |
| 3 | ≥140 kph | <50% | Medium-fast |
| 4 | ≥100 kph | <70% | Medium |
| 5 | ≥60 kph | up to ~100% | Slow (Variante della Roggia, Lesmo) |
| 6 | <60 kph | full | Hairpins (Loews at Monaco, La Source) |

The coach's pace-note vocabulary covers severity words `one` through `six`, so the cached clips are `left one`, `right four`, etc.

## What the coach does with this file

`CoachAgent.fast_tick()` binary-searches the next corner by `brake_point_m`, computes time-to-brake, and at three fixed windows (3s, 1.5s, post-exit) plays a cached pace-note clip via the `Player`. The `name` is used by the slow loop's LLM prompt for human-readable corner references.
