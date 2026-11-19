"""Append-only JSONL flight log.

One JSON object per line, flushed immediately, so a crash or a pulled battery
still leaves a readable trace. Logging failures never interrupt a flight: the
error is kept in :attr:`FlightLogger.error` and shown by the TUI.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path


class FlightLogger:
    """Writes ``{"ts": ..., "event": ..., ...}`` lines to a dated file."""

    def __init__(self, directory: str | Path = "logs", prefix: str = "flight") -> None:
        self.directory = Path(directory).expanduser()
        self.path: Path | None = None
        self.error: str | None = None
        self._lock = threading.Lock()
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.path = self.directory / f"{prefix}_{stamp}.jsonl"
            self.path.touch()
        except OSError as exc:
            self.error = str(exc)

    def log(self, event: str, **fields: object) -> None:
        """Append one event; never raises."""
        if self.path is None:
            return
        record = {"ts": datetime.now().isoformat(timespec="milliseconds"), "event": event}
        record.update(fields)
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
        except (TypeError, ValueError) as exc:
            self.error = str(exc)
            return
        with self._lock:
            try:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except OSError as exc:
                self.error = str(exc)

    def close(self) -> None:
        """Log the end of the session; the file needs no explicit close."""
        self.log("log_closed")


class FlightTrack:
    """Collects position samples so a flight can be replayed as a route.

    The summary is written as a single ``flight_summary`` event and contains
    everything needed to draw the path offline: start/end points, horizontal
    length, duration and how high the drone climbed above its takeoff point.
    """

    def __init__(self) -> None:
        self.started_at = datetime.now()
        self.samples: list[dict[str, float | None]] = []
        self.takeoff_altitude: float | None = None
        self.max_altitude: float | None = None
        self.path_length_m = 0.0
        self._last_position: tuple[float, float, float] | None = None

    def mark_takeoff(self, altitude_m: float | None) -> None:
        self.takeoff_altitude = altitude_m
        self.max_altitude = altitude_m

    def add(self, telemetry) -> None:
        """Append one telemetry sample and update distance/altitude stats."""
        position = telemetry.position
        if telemetry.altitude_m is not None:
            if self.max_altitude is None or telemetry.altitude_m > self.max_altitude:
                self.max_altitude = telemetry.altitude_m
        if position is None:
            return
        x, y, z = (float(value) for value in position)
        if self._last_position is not None:
            px, py, _ = self._last_position
            self.path_length_m += ((x - px) ** 2 + (y - py) ** 2) ** 0.5
        self._last_position = (x, y, z)
        self.samples.append(
            {
                "t": round((datetime.now() - self.started_at).total_seconds(), 2),
                "x": round(x, 3),
                "y": round(y, 3),
                "z": round(z, 3),
                "yaw_deg": None if telemetry.yaw_deg is None else round(telemetry.yaw_deg, 1),
            }
        )

    def altitude_gain(self) -> float | None:
        if self.max_altitude is None or self.takeoff_altitude is None:
            return None
        return max(0.0, self.max_altitude - self.takeoff_altitude)

    def summary(self) -> dict[str, object]:
        first = self.samples[0] if self.samples else {}
        last = self.samples[-1] if self.samples else {}
        return {
            "duration_s": round((datetime.now() - self.started_at).total_seconds(), 1),
            "samples": len(self.samples),
            "takeoff_altitude_m": self.takeoff_altitude,
            "max_altitude_m": self.max_altitude,
            "altitude_gain_m": self.altitude_gain(),
            "path_length_m": round(self.path_length_m, 2),
            "start": {k: first.get(k) for k in ("x", "y", "z")},
            "end": {k: last.get(k) for k in ("x", "y", "z")},
        }
