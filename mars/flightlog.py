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
