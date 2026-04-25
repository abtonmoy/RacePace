# RacePace

Race telemetry AI. Three agents that watch a sim session and talk to the driver — a post-session **analyst**, a real-time **race engineer**, and a real-time **driving coach** — across multiple sims, with voice output, plus end-to-end pipelines that consume real F1 telemetry from FastF1 or OpenF1.

```
        ┌──────────────┐         ┌──────────────────┐
sim ───▶│  Adapter     │  frames │  Ring buffer +   │ snapshot ┌─────────────┐
        │  (ACC/F1/    │────────▶│  SQLite writer   │─────────▶│ Engineer    │── radio
        │   SimHub/...) │         └──────────────────┘          │ (~2s tick)  │
        └──────────────┘                  │                     └─────────────┘
                                          │ snapshot            ┌─────────────┐
                                          ├────────────────────▶│ Coach       │── pace notes
                                          │                     │ fast (30Hz) │   + sector
                                          │                     │ slow (per   │     coaching
                                          │                     │  sector)    │
                                          │                     └─────────────┘
                                          ▼
                                  ┌──────────────┐              ┌─────────────┐
                                  │ session.db   │ ───────────▶ │ Analyst     │── debrief
                                  │ (sqlite)     │  post-race   │ (one-shot)  │
                                  └──────────────┘              └─────────────┘
```

## The three agents

| Agent | Loop | Latency budget | Job |
|---|---|---|---|
| **Analyst** (`agents/analyst.py`) | One-shot, post-session | None (offline) | Read a finished session, write a 300–500 word three-section debrief |
| **Engineer** (`agents/engineer.py`) | ~2s tick | Seconds | Strategic radio: pit calls, fuel warnings, position updates, undercut threats. Mostly silent. |
| **Coach** (`agents/coach.py`) | Two loops: ~30Hz fast + per-sector slow | <150ms (fast loop) | Fast: cached pace-note callouts ("left three, brake"). Slow: one targeted coaching sentence between corners. |

All three use **Gemini (`gemini-2.5-pro` for the analyst, `gemini-2.5-flash` for the live agents)**. Gemini's adaptive thinking is **disabled** for the live agents because we want full max-tokens budget to go to output, not deliberation.

## Quickstart

You need Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install (cross-platform, includes mock adapter for development)
uv sync --extra dev

# Optional extras: see the table below
uv sync --extra dev --extra acc            # on the Windows rig that runs ACC
uv sync --extra dev --extra fastf1         # to import real F1 telemetry from FastF1
uv sync --extra dev --extra voice          # for real audio playback (sounddevice)

# API key for the LLM agents
export GEMINI_API_KEY=your-key-from-aistudio.google.com
```

### One-minute end-to-end on real F1 data (no sim required)

```bash
# Pull Verstappen's first 15 laps of the 2023 Italian GP
uv run racepace openf1-import \
  --year 2023 --country Monza --session-name Race \
  --driver-number 1 --output ver_monza.db --lap-lo 1 --lap-hi 15

# Post-session analyst — writes a markdown debrief to stdout
uv run racepace analyze ver_monza.db

# Build a track map and a reference lap (auto-extracted from VER's lap 3)
uv run racepace extract-track ver_monza.db --lap 3 \
  --output racepace/tracks/f1/monza.json --track monza
uv run racepace save-reference ver_monza.db --lap 3 \
  --track monza --car "Red Bull Racing"
```

The engineer and coach are designed to run live (against a sim adapter or a mock replay).
See the per-agent docs in `racepace/agents/README.md`.

### One-minute end-to-end on a real ACC session

On Windows with ACC running:

```bash
uv run racepace coach --sim acc --output session.db          # records + engineer + coach
uv run racepace analyze session.db                            # post-session debrief
```

`racepace coach` is the production command — it spawns the producer (adapter → ring + writer)
plus the engineer and coach threads, runs until Ctrl-C, and (by default) runs the analyst on
the recorded session afterward.

## Install (extras)

| Extra | What it adds | When to install |
|---|---|---|
| `dev` | pytest, pytest-cov | Always (running the test suite) |
| `acc` | `pyaccsharedmemory` | Windows rig that runs ACC |
| `voice` | `sounddevice` | Real audio playback (TTS clips, slow-loop coaching) |
| `simhub` | `websocket-client` | Windows + SimHub, for sims without a native adapter |
| `fastf1` | `fastf1` | Importing real F1 telemetry via FastF1 |

`uv sync` reads `pyproject.toml` directly; you don't manage `.venv` manually. Combine extras with `--extra a --extra b`.

## CLI reference

All commands live under `racepace`. Run `uv run racepace --help` for the live list.

### Recording and replay

```bash
racepace record --sim {acc|f1|simhub|mock} --output session.db
  [--hz 30] [--replay-path PATH] [--replay-speed 1.0]
```

Streams frames from the chosen adapter into a SQLite session DB until Ctrl-C. With `--sim mock --replay-path other.db`, replays a previously-recorded session at real time (or accelerated via `--replay-speed`).

### Analysis (Phase 1)

```bash
racepace analyze session.db [--session-id ...] [--save debrief.md] [--print-report]
```

Loads the session, builds a structured situation report (pure functions over the frames), and asks Gemini for a three-section markdown debrief. The model is forbidden from inferring numbers — every figure comes verbatim from the report dict.

### Live engineer (Phase 2)

```bash
racepace engineer --sim {acc|f1|simhub|mock} --output session.db [--log out.txt]
```

Spawns producer + engineer threads. The engineer ticks every 2 seconds, evaluates triggers (pit window, fuel margin, position change, undercut threat, periodic), and uses `gemini-2.5-flash` for terse radio calls. Always records to disk so the analyst can review afterward.

### Live coach (Phase 3 — the production command)

```bash
racepace coach --sim {acc|f1|simhub|mock} --output session.db
  [--track-map PATH]                       # default: racepace/tracks/{sim}/{track}.json
  [--references-root references/]          # default lookup: {sim}/{track}/{car}.parquet
  [--voice/--no-voice]                     # enable TTS (requires --extra voice)
  [--analyst-on-finish/--no-analyst]       # run analyst on Ctrl-C
```

Runs everything: producer + engineer + coach (fast + slow loops) + (optional) analyst-on-finish.
Falls back to "pace-notes-only" mode if no track map is found, and to "no comparative coaching"
if no reference lap is found — never refuses to start.

### Track map and reference

```bash
racepace extract-track session.db --lap N --output tracks/{sim}/{track}.json [--track NAME]
racepace save-reference session.db --lap N [--track NAME] [--car CAR]
                                  [--output PATH | --references-root references/]
```

`extract-track` runs the corner extractor on lap N (resample to 1m, find local speed minima,
walk back to brake point, walk forward to exit, derive direction from steering, score severity
1–6). The output JSON is meant to be hand-edited to add corner names and notes.

`save-reference` writes the lap as a Parquet on a 1m grid for the coach's `live_delta` and
`sector_delta` functions.

### Real F1 telemetry importers

```bash
# FastF1 — uses the official F1 + Ergast servers, has a local cache, full Python API
racepace fastf1-import --year YYYY --gp NAME --session-type {R|Q|FP1|FP2|FP3|Sprint}
                       --driver TLA --output session.db [--lap-lo N --lap-hi M]

# OpenF1 — public REST API at openf1.org, stdlib HTTP client, finer-grained position data
racepace openf1-import --year YYYY --country NAME --session-name {Race|Qualifying|...}
                       --driver-number N --output session.db [--lap-lo N --lap-hi M]
```

See `racepace/data/README.md` for the FastF1 vs OpenF1 trade-offs.

## Architecture

Four principles guide the codebase:

1. **Adapters know about sims; agents do not.** Anything sim-specific lives in `adapters/`. Agents only import from `schema.py`.
2. **Normalize aggressively.** Convert to SI (kph, kg, °C, m, s, psi) inside the adapter. Agents assume one unit system.
3. **Missing data is `None`, not zero.** Different sims expose different fields. The agents handle `None` gracefully.
4. **Storage format is the schema.** What you write to disk is the same `TelemetryFrame` shape the agents consume in memory.

Threading:
- Producer thread: adapter → `RingBuffer.push` + `SessionWriter.write_frame`
- Consumer threads: each agent reads `RingBuffer.snapshot()` on its tick interval
- One lock (the ring buffer's). No multiprocessing — the data volume is trivial and shared-memory adapters need to live in the main process on Windows.

The threading model is simple but easy to get wrong. **If you find yourself adding more locks, stop and rethink.**

### Subsystem READMEs

| Path | Read it for |
|---|---|
| `racepace/adapters/README.md` | The adapter contract, how to add a new sim |
| `racepace/agents/README.md` | The three-agent design, the two-loop coach, trigger logic |
| `racepace/voice/README.md` | Voice layer: clip cache, TTS backends, audio player, latency model |
| `racepace/data/README.md` | FastF1 vs OpenF1 — endpoints, units, gotchas |
| `racepace/tracks/README.md` | Track-map JSON format and the hand-edit step |

## Schema reference

`racepace/schema.py`. Bump `SCHEMA_VERSION` (currently `"1.0.0"`) on any breaking change; the storage layer refuses to load files written under an incompatible major version.

### `TelemetryFrame`

| Field | Type | Required | Notes |
|---|---|---|---|
| `timestamp_s` | float | yes | Monotonic, seconds from session start |
| `lap_number` | int | yes | 1-indexed; 0 for out-lap before SF |
| `lap_distance_m` | float / None | | Distance from SF along racing line |
| `lap_distance_pct` | float / None | | 0.0–1.0 of current lap |
| `speed_kph` | float / None | | Ground speed |
| `throttle_pct` | float / None | | 0–100 |
| `brake_pct` | float / None | | 0–100 (some sims expose only on/off → 0 or 100) |
| `clutch_pct` | float / None | | 0–100 (0 = released) |
| `steering_norm` | float / None | | -1 (full left) to +1 (full right) |
| `gear` | int / None | | -1 reverse, 0 neutral, 1+ forward |
| `rpm` | float / None | | |
| `fuel_kg` | float / None | | Mass, not litres. F1 broadcast doesn't expose this. |
| `fuel_laps_remaining` | float / None | | |
| `tyre_wear_pct` | dict / None | | Keys `fl, fr, rl, rr`. Worn %, not remaining. |
| `tyre_temp_c` | dict / None | | Same keys |
| `tyre_pressure_psi` | dict / None | | Same keys |
| `tyre_compound` | str / None | | Sim-specific (e.g. `"dry"`, `"medium"`) |
| `position` | int / None | | 1-indexed |
| `gap_ahead_s` | float / None | | Sim-dependent — often only available on F1 broadcast |
| `gap_behind_s` | float / None | | |
| `last_lap_time_s` | float / None | | Sim's reported last-lap |
| `best_lap_time_s` | float / None | | Sim's reported best-lap |
| `current_sector` | int / None | | 0-indexed |
| `extras` | dict | | Sim-specific richness (g-forces, brake bias, ERS, DRS, …) |

### `SessionInfo`

| Field | Type | Required | Notes |
|---|---|---|---|
| `session_id` | str | yes | UUID |
| `sim` | `"acc"\|"f1"\|"iracing"\|"simhub"` | yes | |
| `started_at` | datetime | yes | |
| `track`, `car`, `session_type`, `weather` | str / None | | |
| `track_temp_c`, `air_temp_c` | float / None | | |
| `total_laps` | int / None | | |
| `ended_at` | datetime / None | | |

### `SituationReport` (live agents)

The Phase 2 / Phase 3 agents consume a `SituationReport` (`features/situation.py`) built fresh from a snapshot on every tick. Field names are a public API — renaming is painful. See `racepace/agents/README.md` for the full layout.

## Configuration

| Env var | Used by | Notes |
|---|---|---|
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | Analyst, Engineer, Coach | Required for any LLM-driven agent |
| `PIPER_MODEL_PATH` | Voice layer | If set and the file exists, `default_tts()` uses Piper TTS |

Cache directories (auto-created, gitignored):

| Dir | What lives there |
|---|---|
| `.fastf1_cache/` | FastF1's downloaded telemetry/timing data |
| `voice_cache/` | Pre-rendered TTS WAVs (cached pace-note phrases) |
| `references/` | Auto-saved reference-lap Parquets per sim/track/car |

## Development

```bash
uv sync --extra dev
uv run pytest -q                                # full test suite (84 tests)
uv run python -m racepace.tests.fixtures.build_fixture            # rebuild ACC fixture
uv run python -m racepace.tests.fixtures.build_track_fixtures     # rebuild track-map fixtures
```

Per-subsystem extension guides live in the subsystem READMEs (above).

## Tips and gotchas

- **Brake from F1 broadcast (FastF1, OpenF1) is binary** — 0 or 100, no pressure. The brake-point detector still works because we look for the first non-zero brake.
- **Steering from F1 broadcast is not exposed.** Both real-data importers derive a heading-delta-of-X/Y proxy. The L/R sign sometimes flips depending on the coordinate convention; corner positions are correct.
- **OpenF1 X/Y is in decimeters** (1/10 m) — not meters. The OpenF1 importer compensates.
- **OpenF1 `/v1/position` only logs position changes.** Use most-recent-at-or-before semantics, not nearest-in-time, or you'll grab future overtakes.
- **`is_clean` heuristic is rate-per-second** (kph/s), so it works across sample rates (F1 ~4 Hz, ACC ~30 Hz). Threshold 250 kph/s — well above peak F1 braking, well below an off-track discontinuity.
- **Tests run faster than wall-clock**, so all suppression timers in the live agents (engineer's 15s gap, coach's 0.8s callout gap) use **session time** (latest frame's `timestamp_s`), not `time.monotonic()`.
- **Live audio (`sounddevice`) is optional.** Without it, the player falls back to `NullBackend` (no-op) — pipeline still runs end-to-end, no audio.
- **`gemini-2.5-flash` thinking is disabled** for the live agents (`thinking_budget=0`) — for one-sentence radio calls deliberation just eats the output budget and truncates.
# RacePace
