"""Pixel-to-metre scale estimated from ArUco markers of known size.

Implements step B2 of ``docs/stage2-plan.md``: the marker side in pixels divided
by its physical side gives ``px_per_meter`` at the flight altitude. The median
over all detected markers rejects a single badly localised corner.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable

from mars.aruco import Marker


def px_per_meter(
    markers: Iterable[Marker] | None, marker_size_m: float
) -> float | None:
    """Return median pixels per metre, or ``None`` when no marker is usable.

    Raises :class:`ValueError` when ``marker_size_m`` is not positive.
    """
    if marker_size_m <= 0:
        raise ValueError("marker_size_m must be positive")

    sides = [marker.side_px for marker in markers or () if marker.side_px > 0]
    if not sides:
        return None
    return statistics.median(sides) / marker_size_m
