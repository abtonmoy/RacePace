# Adapters

The adapter is the boundary between a sim and the rest of RacePace. **Adapters know about sims; agents do not.** Every adapter emits the same `TelemetryFrame` shape, in SI units, regardless of what the underlying sim sends.

## Contract

`AbstractAdapter` (`base.py`):

```python
class AbstractAdapter(ABC):
    sim_name: str            # "acc", "f1", "simhub", ...
    target_hz: float = 30.0  # downsample to this rate (don't interpolate up)

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def read_session_info(self) -> SessionInfo: ...
    def stream_frames(self) -> Iterator[TelemetryFrame]: ...
```

Two responsibilities the adapter owns:

1. **Unit conversion.** Convert sim-native units to SI before emitting (kph, kg, °C, m, s, psi). Examples:
   - ACC reports fuel in litres → multiply by `_LITRES_TO_KG = 0.75`
   - F1 packs throttle as 0.0–1.0 → multiply by 100
   - OpenF1 X/Y is decimeters → divide by 10 in the cumulative distance integration

2. **Resampling.** If the source runs faster than `target_hz`, downsample using `RateLimiter` from `base.py`. If slower, **emit at native rate and let consumers handle gaps** — do not interpolate up.

## What's here

| File | Purpose |
|---|---|
| `base.py` | `AbstractAdapter` + `RateLimiter` |
| `acc.py` | ACC shared memory adapter (Windows-only). Per-field mapping in the docstring. |
| `f1.py` | F1 24/25 UDP telemetry adapter. Stdlib socket + struct, no external deps. |
| `simhub.py` | SimHub WebSocket fallback. Optional `websocket-client` dep. Windows-only. |
| `mock.py` | Replays a recorded SQLite session at real-time pace (or accelerated). |

## Adding a new adapter

1. Subclass `AbstractAdapter` in `racepace/adapters/<sim>.py`.
2. Implement the four methods. Lazily import any sim-specific package inside `connect()` so the package stays optional.
3. Add a per-field docstring listing every `TelemetryFrame` field and how it's derived (or noted as unavailable).
4. Wire it into `racepace/cli/record.py::_build_adapter`.
5. Add `pyproject.toml` extra if it needs a new dep. Mirror the `[acc]` / `[simhub]` shape.

## Field-by-field source maps

Each adapter's docstring documents which sim field maps to each `TelemetryFrame` field, with notes on what the sim doesn't expose. Read those docstrings before assuming a field is available — F1 broadcast doesn't give you fuel mass, ACC SHM doesn't give you precise gap-to-leader, etc.

## Choosing between adapters when more than one works for a sim

- **Native > SimHub.** SimHub is the long tail. ACC has `acc.py`; use that. F1 24/25 has `f1.py`; use that.
- **Live > replay.** Use `mock.py` for development on macOS/Linux, or for replaying a real session through the engineer/coach offline.
