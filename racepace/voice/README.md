# Voice layer

Three pieces, one drop-in sink.

| File | Role |
|---|---|
| `player.py` | Non-blocking audio queue with pluggable backends |
| `cache.py` | In-RAM clip cache for the coach's pace-note vocabulary |
| `live_tts.py` | TTS backends + on-disk hash-memo decorator |
| `voice_out.py` | `VoiceOutput` — drop-in for `comms.text_out.TextOutput` (same `.send(msg)` signature) |

## Why two paths (cache vs TTS)?

The coach's fast loop has a <150ms latency budget. A live TTS API call is 500ms+. So:

- **Pre-cache** every phrase the fast loop is allowed to say (`left three`, `brake`, `flat`, etc.) — synthesized once at startup, kept as `np.float32` in RAM, played with zero IO on the hot path.
- **Live TTS** for the slow loop's coaching sentences and the engineer's radio calls — slow path, 500ms+ of latency is fine because both fire between corners.

`build_default_cache(tts, cache_dir)` synthesizes the default vocabulary if not already on disk, then loads from disk on subsequent runs. `OnDiskCachedTTS` wraps any backend to hash-memo by SHA1 of `(voice_id, text)`.

## Audio backend

`AudioBackend` protocol with three implementations:

| Backend | When | Notes |
|---|---|---|
| `SoundDeviceBackend` | Production audio | Lazy-imports `sounddevice` (PortAudio). Fall back if unavailable. |
| `NullBackend` | No audio HW (CI, headless rigs) | Sleeps the clip's duration so timing is realistic |
| `RecordingBackend` | Tests | Records every play into a list, no sleep |

`Player.play(samples, sr)` returns immediately. A worker thread drains the queue. Clips queued more than `max_age_s` ago (default 500ms) get dropped — **a stale callout is worse than no callout.**

## TTS backend

`TTSBackend` protocol with three implementations:

| Backend | When | Notes |
|---|---|---|
| `PiperTTSBackend` | Production (offline, free) | Subprocess to local `piper` binary. Set `PIPER_MODEL_PATH` env var. |
| `NullTTSBackend` | No TTS configured | Returns short silent buffers — pipeline still runs, just inaudible |
| `RecordingTTSBackend` | Tests | Captures requested texts |

`default_tts()` picks Piper if `PIPER_MODEL_PATH` is set and the model file exists, else `NullTTSBackend`.

## Latency model (fast loop)

```
frame arrives ──▶ ring.push  ──▶ fast_tick  ──▶ player.play(samples)
~250ms apart      O(1) lock     <1ms compute    O(1) queue.put_nowait
                                                       │
                                                       └─▶ worker thread ──▶ backend.play
                                                                                │
                                                                                └─▶ audio (~50ms PortAudio)
```

End-to-end frame→audio is dominated by `backend.play` (PortAudio's internal buffering, typically 30–50ms). The Python side is well under 5ms.

## VoiceOutput

`VoiceOutput.send(msg)` is shape-compatible with `TextOutput.send(msg)`, so any agent that takes an `output` argument works with either. It:

1. Prints `[HH:MM:SS] PREFIX: msg` to stdout (same as TextOutput)
2. Writes to optional log file
3. Spawns a daemon thread to synthesize TTS off the caller's thread (so a slow TTS never blocks the agent loop)
4. Queues the synthesized samples into the shared `Player`

The player is shared between the engineer's voice output and the coach's slow-loop voice output — two voice sources, one mixed audio queue, no overlapping playback.
