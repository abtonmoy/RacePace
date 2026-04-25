"""Unit tests for every strategy function with hand-computed expected values."""

from __future__ import annotations

import math

from racepace.features import strategy
from racepace.features.laps import Lap


def _lap(num, lap_time=90.0, fuel_used=2.5, wear=None, clean=True, complete=True):
    return Lap(
        lap_number=num,
        frames=[],
        lap_time_s=lap_time,
        is_clean=clean,
        is_complete=complete,
        fuel_used_kg=fuel_used,
        tyre_wear_delta_pct=wear,
    )


# --- pit_loss_seconds ---------------------------------------------------------

def test_pit_loss_known_track():
    assert strategy.pit_loss_seconds("Spa", "acc") == 23.0


def test_pit_loss_unknown_track_falls_back():
    assert strategy.pit_loss_seconds("Wherever", "acc") == strategy.DEFAULT_PIT_LOSS_S


def test_pit_loss_no_track_falls_back():
    assert strategy.pit_loss_seconds(None) == strategy.DEFAULT_PIT_LOSS_S


# --- fuel_per_lap -------------------------------------------------------------

def test_fuel_per_lap_window_three():
    laps = [_lap(i, fuel_used=2.0 + 0.1 * i) for i in range(1, 6)]
    # Last 3 = 2.3, 2.4, 2.5; mean = 2.4
    assert strategy.fuel_per_lap(laps) == 2.4


def test_fuel_per_lap_skips_dirty():
    laps = [_lap(1, fuel_used=2.5), _lap(2, fuel_used=10.0, clean=False), _lap(3, fuel_used=2.5)]
    # Dirty lap 2 is excluded
    assert strategy.fuel_per_lap(laps) == 2.5


def test_fuel_per_lap_skips_refuel_negative():
    laps = [_lap(1, fuel_used=-50.0), _lap(2, fuel_used=2.5)]
    assert strategy.fuel_per_lap(laps) == 2.5


def test_fuel_per_lap_returns_none_when_no_data():
    laps = [_lap(1, fuel_used=None)]
    assert strategy.fuel_per_lap(laps) is None


# --- fuel_margin_laps ---------------------------------------------------------

def test_fuel_margin_laps_basic():
    # 25 kg / 2.5 kg/lap = 10 laps; finish in 8 = margin 2.0
    assert strategy.fuel_margin_laps(25.0, 2.5, 8) == 2.0


def test_fuel_margin_laps_negative_means_short():
    # 5 kg / 2.5 = 2 laps; need 5 more; margin = -3
    assert strategy.fuel_margin_laps(5.0, 2.5, 5) == -3.0


def test_fuel_margin_laps_none_inputs():
    assert strategy.fuel_margin_laps(None, 2.5, 5) is None
    assert strategy.fuel_margin_laps(10, None, 5) is None
    assert strategy.fuel_margin_laps(10, 2.5, None) is None
    assert strategy.fuel_margin_laps(10, 0.0, 5) is None  # divide-by-zero guard


# --- laps_to_finish -----------------------------------------------------------

def test_laps_to_finish_lap_counted():
    assert strategy.laps_to_finish(total_laps=20, current_lap=8) == 12


def test_laps_to_finish_past_end():
    assert strategy.laps_to_finish(total_laps=20, current_lap=21) == 0


def test_laps_to_finish_time_based_with_avg():
    # 600s remaining / 90s per lap = 6.67 → ceil = 7, plus extra 1 = 8
    assert strategy.laps_to_finish(
        total_laps=None, current_lap=10, avg_lap_time_s=90.0, session_time_remaining_s=600.0
    ) == 8


def test_laps_to_finish_no_signal():
    assert strategy.laps_to_finish(total_laps=None, current_lap=5) is None


# --- tyre_wear_per_lap --------------------------------------------------------

def test_tyre_wear_per_lap_avg_over_corners_and_laps():
    # Each lap: avg of [1, 1, 1, 1] = 1.0; window=3 returns 1.0
    laps = [
        _lap(i, wear={"fl": 1.0, "fr": 1.0, "rl": 1.0, "rr": 1.0}) for i in range(1, 5)
    ]
    assert strategy.tyre_wear_per_lap(laps) == 1.0


def test_tyre_wear_per_lap_skips_negative_delta():
    # Lap 2 has negative delta = tyre change; excluded
    laps = [
        _lap(1, wear={"fl": 1.0, "fr": 1.0, "rl": 1.0, "rr": 1.0}),
        _lap(2, wear={"fl": -50.0, "fr": -50.0, "rl": -50.0, "rr": -50.0}),
        _lap(3, wear={"fl": 1.5, "fr": 1.5, "rl": 1.5, "rr": 1.5}),
    ]
    assert strategy.tyre_wear_per_lap(laps) == 1.25  # mean of 1.0 and 1.5


# --- laps_until_critical_wear -------------------------------------------------

def test_laps_until_critical_basic():
    # 50% worn now, 1.0%/lap, critical at 80% → 30 laps
    assert strategy.laps_until_critical_wear(50.0, 1.0) == 30.0


def test_laps_until_critical_already_past():
    assert strategy.laps_until_critical_wear(85.0, 1.0) == 0.0


def test_laps_until_critical_no_data():
    assert strategy.laps_until_critical_wear(None, 1.0) is None
    assert strategy.laps_until_critical_wear(50.0, None) is None
    assert strategy.laps_until_critical_wear(50.0, 0.0) is None


# --- undercut_advantage -------------------------------------------------------

def test_undercut_advantage_classic():
    # Fresh tyres do 89s, ahead is doing 90s → undercutter gains 1s/lap
    assert strategy.undercut_advantage(89.0, 90.0) == 1.0


def test_undercut_advantage_with_pit_diff():
    # Same pace gain, but you lose 0.5s extra in the pits
    assert strategy.undercut_advantage(89.0, 90.0, pit_loss_diff_s=0.5) == 0.5


# --- is_undercut_threat -------------------------------------------------------

def test_undercut_threat_within_pit_loss_and_faster():
    # Behind is 2.5s back, both currently doing 90s, pit loss 22s, fresh gain 1.5s.
    # Behind on fresh = 88.5s vs us at 90s → threat.
    assert strategy.is_undercut_threat(
        gap_behind_s=2.5, self_pace_old_s=90.0, behind_pace_old_s=90.0, pit_loss_s=22.0
    ) is True


def test_undercut_threat_too_far_back():
    assert strategy.is_undercut_threat(
        gap_behind_s=30.0, self_pace_old_s=90.0, behind_pace_old_s=90.0, pit_loss_s=22.0
    ) is False


def test_undercut_threat_behind_is_slower():
    # Behind is doing 92s, fresh = 90.5 — still slower than us at 90 → not a threat.
    assert strategy.is_undercut_threat(
        gap_behind_s=2.0, self_pace_old_s=90.0, behind_pace_old_s=92.0, pit_loss_s=22.0
    ) is False


# --- is_overtake_opportunity --------------------------------------------------

def test_overtake_opportunity_close_and_faster():
    assert strategy.is_overtake_opportunity(
        gap_ahead_s=1.0, self_pace_s=89.5, ahead_pace_s=90.5
    ) is True


def test_overtake_opportunity_too_far_ahead():
    assert strategy.is_overtake_opportunity(
        gap_ahead_s=5.0, self_pace_s=89.5, ahead_pace_s=90.5
    ) is False


# --- pit_window_open ----------------------------------------------------------

def test_pit_window_open_on_low_fuel_margin():
    assert strategy.pit_window_open(fuel_margin=2.0, laps_until_critical_wear_count=20) is True


def test_pit_window_open_on_imminent_tyre_critical():
    assert strategy.pit_window_open(fuel_margin=10.0, laps_until_critical_wear_count=4) is True


def test_pit_window_closed_when_neither():
    assert strategy.pit_window_open(fuel_margin=10.0, laps_until_critical_wear_count=20) is False


def test_pit_window_closed_when_no_signal():
    assert strategy.pit_window_open(None, None) is False


# --- optimal_pit_lap ----------------------------------------------------------

def test_optimal_pit_lap_no_constraints_returns_none():
    assert strategy.optimal_pit_lap(
        current_lap=10,
        laps_to_finish_count=20,
        fuel_margin=10.0,
        laps_until_critical_wear_count=None,
    ) is None


def test_optimal_pit_lap_tyre_constrained():
    # Tyres die in 5 laps; minus 1-lap safety buffer → pit at lap 14
    assert strategy.optimal_pit_lap(
        current_lap=10,
        laps_to_finish_count=20,
        fuel_margin=10.0,
        laps_until_critical_wear_count=5.0,
    ) == 14


def test_optimal_pit_lap_fuel_constrained():
    # Fuel margin = -3 means 3 laps short; total fuel laps left = 20 - 3 = 17;
    # minus safety buffer 1 = 16 more laps; pit at lap 10 + 16 = 26
    # (Caller can sanity-check vs laps_to_finish.)
    assert strategy.optimal_pit_lap(
        current_lap=10,
        laps_to_finish_count=20,
        fuel_margin=-3.0,
        laps_until_critical_wear_count=None,
    ) == 26


def test_optimal_pit_lap_box_now_when_constraint_already_violated():
    assert strategy.optimal_pit_lap(
        current_lap=10,
        laps_to_finish_count=20,
        fuel_margin=10.0,
        laps_until_critical_wear_count=0.5,
    ) == 10
