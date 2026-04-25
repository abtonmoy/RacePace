# Agents

Three agents, three completely different latency budgets and design philosophies.

| Agent | File | Loop | LLM | Latency budget | Sim-specific knowledge? |
|---|---|---|---|---|---|
| Analyst | `analyst.py` | One-shot, post-session | `gemini-2.5-pro` (adaptive thinking on) | None — runs offline | None |
| Engineer | `engineer.py` | ~2s tick | `gemini-2.5-flash` (thinking off) | Seconds | None |
| Coach (slow) | `coach.py` | Per-sector | `gemini-2.5-flash` (thinking off) | A few seconds | Track map + reference lap |
| Coach (fast) | `coach.py` | ~30Hz | None — cached audio clips | <150ms frame→audio | Track map |

## Architectural rules they all share

1. **Numbers come from the report, not the model.** LLMs are bad at arithmetic. Every figure the agent says comes verbatim from a structured situation report built by pure functions. The system prompts forbid the model from inferring values.
2. **Suppression timers run on session time, not wall-clock.** Tests and replays run faster than real time. Wall-clock suppression silently breaks both. All `*_session_t` fields use the latest frame's `timestamp_s`.
3. **One-shot LLM calls. No tools, no agentic loops.** Even the slow loop in the coach is a single `client.models.generate_content(...)` call.

## Analyst

Loads a finished session via `SessionReader`, runs the feature extractors (`features/laps`, `features/deg`), builds a structured **situation report** dict, and asks `gemini-2.5-pro` for a 300–500 word three-section debrief: *what went well, where time was lost, what to work on*.

The debrief is the only thing the LLM sees with adaptive thinking enabled — it's an offline analytical task where careful reasoning is fine and quality matters more than speed.

## Engineer

Lives between corners. Strategic radio: pit calls, fuel warnings, position updates, undercut threats. **The biggest mistake is a chatty engineer.** A real race engineer says ~30 things in a 1-hour race; ours should too.

The trigger state machine in `engineer.py::should_speak`:

| Trigger | Class | When it fires |
|---|---|---|
| `fuel_critical` | one-shot, **bypasses suppression** | `fuel_margin_laps < 0.5` |
| `tyre_critical` | one-shot, bypasses suppression | `avg_tyre_wear_pct >= 80` |
| `pit_window_open` | one-shot | `pit_window_open` becomes True |
| `rain_incoming` | one-shot | `rain_incoming` becomes True |
| `fuel_low` | one-shot | `fuel_margin_laps < 1.0` |
| `undercut_threat` | one-shot per occurrence (resets when False) | `undercut_threat` becomes True |
| `position_change_X_to_Y` | every change | Position field changes |
| `overtake_opportunity` | every tick (rare) | `overtake_opportunity` is True |
| `periodic_status` | every 5 laps | Counts laps since last periodic |

15-second session-time suppression between non-critical messages.

The engineer's prompt tells Gemini what the trigger means and instructs it to round numbers to one decimal, use no salutation, and output the literal token `NO_COMMENT` if nothing actionable is in the report.

### Verifying the no-chatter rule

`racepace/tests/test_engineer.py::test_no_chatter_in_uneventful_30_lap_stint` hard-asserts the engineer says fewer than 10 things over a 30-lap uneventful stint. Run `uv run pytest racepace/tests/test_engineer.py -v` if you change the trigger logic.

## Coach (two loops)

The hardest agent to build. The corner-entry callout has to land **before** the corner — an LLM round-trip plus TTS is too slow to generate "brake now" in real time. So:

### Fast loop (`fast_tick`, ~30Hz, no LLM)

1. Read `ringbuffer.latest()`.
2. Binary-search the next corner from a precomputed sorted list.
3. Compute time-to-brake = `(corner.brake_point_m - current.lap_distance_m) / (current.speed_kph / 3.6)`.
4. Trigger windows:
   - `3.0–2.6s` to brake point: announce direction + severity ("left, three") via cached clip
   - `1.5–1.1s` to brake point: announce action ("brake" or "flat" or "lift")
   - Past exit point with reference loaded: maybe "carry more speed" if apex was 5+ kph below target
5. Hot path is pure arithmetic + a `dict.get()` on the in-RAM clip cache. No IO, no LLM, no allocation beyond a list comprehension.

Per-(corner_id, kind) dedupe means each corner produces at most one of each callout per lap. 0.8-second session-time min-gap between any two callouts (audio clips overlap-mute each other otherwise).

### Slow loop (`slow_tick`, per sector)

When the driver crosses into a new sector, compute `sector_delta(reference, just_completed_lap, completed_sector)`. If the sector cost more than `0.15s` AND the driver is currently on a straight, ask `gemini-2.5-flash` for one targeted coaching sentence. Quote numbers verbatim from the structured payload.

### Degraded modes

The coach starts even if you're missing data:

- **No track map for this sim+track**: `pace-notes-only mode = True implicit` — fast loop is a no-op (no corners to look up), slow loop early-returns.
- **No reference lap**: Force `pace-notes-only=True` — fast loop still does pace-note callouts (direction + severity from the track map), slow loop early-returns.
- **Both missing**: just records to disk; still useful as input for the analyst afterward.

## Two patterns they all share

### Driver loop (sync, for tests)

The agents expose `build_report()` / `should_speak()` / `generate_message()` (engineer) and `fast_tick()` / `slow_tick()` (coach) as public methods. Tests drive them synchronously instead of starting threads — deterministic, no real-time dependence. Look at `tests/test_engineer.py::_drive` and `tests/test_coach.py::_drive_through` for the pattern.

### LLM call factories

Each agent constructs its `_default_llm_call(api_key)` closure. The closure is a `Callable[[str, str], str]` (engineer) or `Callable[[str], str]` (coach). Tests pass a mock callable so they don't hit the network; production passes the default. Same shape — one easy injection point.
