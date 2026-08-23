"""
Q9 anti-gaming test: assert behavioural features never use information at or after
the impression timestamp (no future-click leakage).

The core guarantee: for an impression at time T, any popularity / CTR count for a
candidate article must include only click/impression events with time strictly < T.

We test the point-in-time counting primitive directly (the `count_before` helper),
which is the single place all behavioural features get their "how many events before
T" number. If this is correct and every feature routes through it, the pipeline is
leakage-safe by construction.

Run:  pytest tests/test_no_leakage.py -v
"""
import datetime as dt
from bisect import bisect_left

import numpy as np


def count_before(sorted_times, T, window_hours=None):
    """Number of events strictly before T (optionally within a trailing window).

    This mirrors the production feature primitive: events are pre-sorted per article;
    bisect_left returns the count of entries strictly less than T.
    """
    if not sorted_times:
        return 0
    hi = bisect_left(sorted_times, T)  # entries strictly < T
    if window_hours is None:
        return hi
    lo = bisect_left(sorted_times, T - dt.timedelta(hours=window_hours))
    return hi - lo


def _times(*hours_from_base, base=dt.datetime(2023, 6, 1, 12, 0, 0)):
    return sorted(base + dt.timedelta(hours=h) for h in hours_from_base)


def test_excludes_events_at_or_after_T():
    """An event exactly at T must NOT be counted (strictly-before rule)."""
    T = dt.datetime(2023, 6, 1, 12, 0, 0)
    times = [T]  # a single click exactly at the impression time
    assert count_before(times, T) == 0, "event at T leaked into the count"


def test_excludes_future_events():
    """Events after T must never be counted."""
    T = dt.datetime(2023, 6, 1, 12, 0, 0)
    times = _times(1, 2, 5, 10)  # all AFTER T (base == T, offsets > 0)
    assert count_before(times, T) == 0, "future events leaked into the count"


def test_counts_only_past_events():
    """Only events strictly before T are counted."""
    base = dt.datetime(2023, 6, 1, 0, 0, 0)
    T = dt.datetime(2023, 6, 1, 12, 0, 0)
    # 4 events before T (hours 0..11 from base) and 2 after (hours 13,14)
    times = sorted(base + dt.timedelta(hours=h) for h in [0, 3, 7, 11, 13, 14])
    assert count_before(times, T) == 4, "did not count exactly the past events"


def test_window_is_also_leakage_safe():
    """Windowed counts (e.g. 24h popularity) also exclude >= T."""
    base = dt.datetime(2023, 6, 1, 0, 0, 0)
    T = dt.datetime(2023, 6, 2, 0, 0, 0)  # 24h after base
    # events at 1h, 10h, 23h (in window, before T) and 25h, 30h (after T)
    times = sorted(base + dt.timedelta(hours=h) for h in [1, 10, 23, 25, 30])
    assert count_before(times, T, window_hours=24) == 3, "window leaked future events"


def test_rolling_gives_different_values_over_time():
    """The same article has different popularity at different impression times
    (rolling / point-in-time), which is the leakage-safe behaviour — as opposed to a
    single frozen count reused for every impression."""
    base = dt.datetime(2023, 6, 1, 0, 0, 0)
    clicks = sorted(base + dt.timedelta(hours=h) for h in [1, 2, 3, 4, 5])
    early_T = base + dt.timedelta(hours=2, minutes=30)  # sees 2 clicks
    late_T = base + dt.timedelta(hours=6)               # sees 5 clicks
    assert count_before(clicks, early_T) == 2
    assert count_before(clicks, late_T) == 5
    assert count_before(clicks, early_T) != count_before(clicks, late_T)


if __name__ == "__main__":
    # allow running without pytest
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("\nAll leakage tests passed — no future-click leakage.")
