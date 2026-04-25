"""Deterministic race-strategy math. No LLM, no I/O.

Every function here returns numbers the engineer agent will quote to the
driver — they need to be defensible. Each function documents its
assumptions explicitly.

Concepts:
- "Stint": laps run on one set of tyres / one fuel load between pit stops.
- "Pit loss": wall-clock penalty of taking the pit lane vs staying out one
  more lap. Includes pit-lane transit + speed limiter, NOT service time
  (since service time scales with what's done — fuel, tyres, repairs).
- "Fuel margin": (fuel_laps_remaining) - (laps_to_finish). Positive means
  you can finish; below ~1 lap is a real concern (formation lap, traffic
  on pit-out, mistake recovery all eat margin).
"""

from __future__ import annotations

from racepace.features.laps import Lap


# --- Pit-loss table -----------------------------------------------------------
# Seconds lost vs staying out one lap, transit + speed-limiter only. Service
# time is added separately by the caller. Values are reasonable starting
# estimates for ACC GT3 and will be refined per car/track over time.
#
# Keys are lowercased track strings as ACC reports them.
PIT_LOSS_S: dict[str, float] = {
    "spa": 23.0,
    "monza": 22.0,
    "nurburgring": 24.0,
    "silverstone": 23.5,
    "imola": 25.0,
    "barcelona": 21.5,
    "paul_ricard": 22.0,
    "zandvoort": 19.5,
    "brands_hatch": 20.0,
    "hungaroring": 22.0,
    "kyalami": 23.0,
    "mount_panorama": 27.0,
    "laguna_seca": 21.0,
    "suzuka": 24.0,
    "watkins_glen": 22.0,
    "fixture-track": 20.0,  # used by tests
}

DEFAULT_PIT_LOSS_S = 22.0


def pit_loss_seconds(track: str | None, sim: str = "acc") -> float:
    """Look up pit loss (transit + limiter) for a track. Falls back to a sane default.

    Assumption: dry conditions, no traffic, clean entry/exit. In wet conditions or
    under safety car, reduce by ~10–20% — caller's responsibility.
    """
    if not track:
        return DEFAULT_PIT_LOSS_S
    return PIT_LOSS_S.get(track.lower(), DEFAULT_PIT_LOSS_S)


# --- Fuel ---------------------------------------------------------------------

def fuel_per_lap(recent_laps: list[Lap], window: int = 3) -> float | None:
    """Rolling mean fuel-per-lap over the most recent N complete clean laps.

    Excludes:
    - laps with no fuel data
    - laps marked dirty (off-tracks/spins eat fuel inconsistently)
    - laps that gained fuel (refuelling — happens during mid-race pits)

    Returns None if fewer than 1 valid sample is available. Uses up to `window`
    samples; small N is intentional — tyre wear and traffic shift fuel use over
    a stint, so we want a recency-biased estimate.
    """
    valid: list[float] = []
    for lap in reversed(recent_laps):
        if lap.fuel_used_kg is None:
            continue
        if lap.is_clean is False:
            continue
        if lap.fuel_used_kg < 0:  # refuelled
            continue
        valid.append(lap.fuel_used_kg)
        if len(valid) >= window:
            break
    if not valid:
        return None
    return sum(valid) / len(valid)


def fuel_margin_laps(
    fuel_kg: float | None,
    fuel_per_lap_kg: float | None,
    laps_to_finish: int | None,
) -> float | None:
    """How many laps of fuel cushion you have over the race finish.

    Positive means you can finish; <= 0 means you cannot. Convention: a
    margin of 1.0 means one full lap of slack — formation, pit-out, or
    one mistake recovery worth.
    """
    if fuel_kg is None or fuel_per_lap_kg is None or fuel_per_lap_kg <= 0:
        return None
    if laps_to_finish is None:
        return None
    laps_remaining = fuel_kg / fuel_per_lap_kg
    return laps_remaining - laps_to_finish


# --- Race length --------------------------------------------------------------

def laps_to_finish(
    total_laps: int | None,
    current_lap: int,
    avg_lap_time_s: float | None = None,
    session_time_remaining_s: float | None = None,
) -> int | None:
    """Laps the driver still has to complete after the current one.

    For lap-counted sessions: total_laps - current_lap.
    For time-based sessions: ceil(session_time_remaining / avg_lap_time) + 1
    (the +1 covers the lap-after-the-clock-hits-zero rule that most sims use).

    Returns None if neither information path is available.
    """
    if total_laps is not None and total_laps > 0:
        return max(0, total_laps - current_lap)
    if session_time_remaining_s is not None and avg_lap_time_s and avg_lap_time_s > 0:
        import math
        return max(0, int(math.ceil(session_time_remaining_s / avg_lap_time_s)) + 1)
    return None


# --- Tyre life ----------------------------------------------------------------

def tyre_wear_per_lap(recent_laps: list[Lap], window: int = 3) -> float | None:
    """Mean wear-percentage delta per lap (averaged across the four corners).

    Uses tyre_wear_delta_pct from the Lap summary. Returns None if no data.
    Negative deltas (wear going down) are dropped — that's a tyre change.
    """
    deltas: list[float] = []
    for lap in reversed(recent_laps):
        if not lap.tyre_wear_delta_pct:
            continue
        per_corner = list(lap.tyre_wear_delta_pct.values())
        if not per_corner:
            continue
        avg = sum(per_corner) / len(per_corner)
        if avg < 0:  # tyre change between laps
            continue
        deltas.append(avg)
        if len(deltas) >= window:
            break
    if not deltas:
        return None
    return sum(deltas) / len(deltas)


def laps_until_critical_wear(
    avg_wear_pct: float | None,
    wear_per_lap: float | None,
    critical_pct: float = 80.0,
) -> float | None:
    """How many more laps until tyres reach `critical_pct` worn.

    `avg_wear_pct` is the current cross-corner average of *worn* % (0 = new).
    Returns None if we lack rate data; returns 0.0 if already past critical.
    """
    if avg_wear_pct is None or wear_per_lap is None or wear_per_lap <= 0:
        return None
    if avg_wear_pct >= critical_pct:
        return 0.0
    return (critical_pct - avg_wear_pct) / wear_per_lap


# --- Undercut / overcut -------------------------------------------------------

def undercut_advantage(
    self_pace_fresh_s: float,
    ahead_pace_old_s: float,
    pit_loss_diff_s: float = 0.0,
) -> float:
    """Per-lap time gained by undercutting on fresh tyres.

    The classic undercut: car A boxes one lap before car B. While A is on
    fresh and B is on old, A gains (B_pace - A_fresh_pace) per lap of
    overlap. The two cars' pit losses cancel out unless they differ
    materially (different fuel loads, tyre service time, etc.).

    Returns the per-lap pace advantage in seconds. Positive = undercut
    favored. The caller compares this × overlap_laps against the gap to
    decide whether to actually call the pit.

    Args:
        self_pace_fresh_s: expected lap time on freshly-fitted tyres.
        ahead_pace_old_s: current lap time of the car ahead (still on old tyres).
        pit_loss_diff_s: (ours - theirs) pit-time difference; usually 0.
    """
    return ahead_pace_old_s - self_pace_fresh_s - pit_loss_diff_s


def is_undercut_threat(
    gap_behind_s: float | None,
    self_pace_old_s: float | None,
    behind_pace_old_s: float | None,
    pit_loss_s: float,
    fresh_tyre_gain_s: float = 1.5,
) -> bool:
    """True if the car behind can undercut us with a one-stop gain.

    Heuristic: they're a threat if the gap is small enough that pit-loss
    plus their first fresh-tyre lap brings them out ahead of us. Compares
    (their_pace - fresh_tyre_gain) vs our pace.

    `fresh_tyre_gain_s` is how much faster a freshly-tyred lap is than an
    old-tyre lap; default 1.5s is a GT3 rule of thumb.
    """
    if gap_behind_s is None or self_pace_old_s is None or behind_pace_old_s is None:
        return False
    if gap_behind_s > pit_loss_s:
        # They lose more in the pits than they can recover.
        return False
    behind_fresh = behind_pace_old_s - fresh_tyre_gain_s
    return behind_fresh < self_pace_old_s


def is_overtake_opportunity(
    gap_ahead_s: float | None,
    self_pace_s: float | None,
    ahead_pace_s: float | None,
    closing_threshold_s: float = 1.5,
) -> bool:
    """True if the car ahead is within 1.5s and slower than us."""
    if gap_ahead_s is None or self_pace_s is None or ahead_pace_s is None:
        return False
    if gap_ahead_s > closing_threshold_s:
        return False
    return self_pace_s < ahead_pace_s


# --- Pit-window decision -----------------------------------------------------

def optimal_pit_lap(
    current_lap: int,
    laps_to_finish_count: int | None,
    fuel_margin: float | None,
    laps_until_critical_wear_count: float | None,
    pit_safety_buffer: int = 1,
) -> int | None:
    """Latest sensible lap to box, given fuel and tyre constraints.

    Strategy: pit as late as possible to keep the gap to the car behind
    large, but no later than the lap where fuel or tyres force it. Adds a
    one-lap safety buffer to the constraint (so you don't run dry on the
    in-lap).

    Returns None if no constraint is binding (driver can run to flag).
    """
    if laps_to_finish_count is None:
        return None

    constraint_laps_remaining: list[float] = []
    if fuel_margin is not None and fuel_margin < 0:
        # Fuel will run out before flag; latest pit = laps you have on this load.
        # Approximate "laps left on this fuel" = laps_to_finish + fuel_margin.
        constraint_laps_remaining.append(max(0.0, laps_to_finish_count + fuel_margin))
    if laps_until_critical_wear_count is not None:
        constraint_laps_remaining.append(laps_until_critical_wear_count)

    if not constraint_laps_remaining:
        return None

    laps_left = min(constraint_laps_remaining) - pit_safety_buffer
    if laps_left <= 0:
        return current_lap  # box now
    return current_lap + int(laps_left)


def pit_window_open(
    fuel_margin: float | None,
    laps_until_critical_wear_count: float | None,
    fuel_threshold_laps: float = 3.0,
    wear_threshold_laps: float = 5.0,
) -> bool:
    """Pit window opens when pit consideration is in the next few laps.

    Specifically: open if fuel margin is below a 3-lap cushion, or tyres
    will hit critical within 5 laps. Either is sufficient.
    """
    if fuel_margin is not None and fuel_margin < fuel_threshold_laps:
        return True
    if laps_until_critical_wear_count is not None and laps_until_critical_wear_count < wear_threshold_laps:
        return True
    return False
