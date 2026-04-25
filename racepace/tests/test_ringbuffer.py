"""Ring buffer correctness."""

from __future__ import annotations

import threading

from racepace.schema import TelemetryFrame
from racepace.storage.ringbuffer import RingBuffer


def _frame(t, lap=1):
    return TelemetryFrame(timestamp_s=t, lap_number=lap)


def test_push_and_snapshot():
    rb = RingBuffer(capacity_seconds=10, expected_hz=10)
    for i in range(5):
        rb.push(_frame(i))
    snap = rb.snapshot()
    assert [f.timestamp_s for f in snap] == [0, 1, 2, 3, 4]


def test_latest_returns_most_recent():
    rb = RingBuffer(capacity_seconds=10, expected_hz=10)
    assert rb.latest() is None
    rb.push(_frame(0))
    rb.push(_frame(1))
    assert rb.latest().timestamp_s == 1


def test_window_returns_recent_seconds():
    rb = RingBuffer(capacity_seconds=10, expected_hz=10)
    for i in range(20):
        rb.push(_frame(i * 0.5))  # 10s of data at 2Hz
    win = rb.window(seconds=2.0)
    # Last frame at t=9.5; cutoff = 7.5 → frames 7.5, 8.0, 8.5, 9.0, 9.5
    assert [f.timestamp_s for f in win] == [7.5, 8.0, 8.5, 9.0, 9.5]


def test_overflow_drops_oldest():
    rb = RingBuffer(capacity_seconds=1, expected_hz=2)  # max_len = 1*2*1.5 = 3
    for i in range(10):
        rb.push(_frame(i))
    snap = rb.snapshot()
    assert [f.timestamp_s for f in snap] == [7, 8, 9]


def test_concurrent_push_consistency():
    rb = RingBuffer(capacity_seconds=100, expected_hz=100)

    def producer(start):
        for i in range(1000):
            rb.push(_frame(start + i * 0.001))

    threads = [threading.Thread(target=producer, args=(s * 10.0,)) for s in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    snap = rb.snapshot()
    # No data races → correct count of all pushed frames.
    assert len(snap) == 4000
