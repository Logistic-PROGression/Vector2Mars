"""Thin flight adapter so vision code never imports hardware SDKs.

``mars.vision`` only consumes frames. All Pioneer SDK 2 usage is confined to
:class:`PioneerAdapter`, which imports the SDK lazily so the vision pipeline can
run and be tested offline.
"""

from __future__ import annotations

import abc
from pathlib import Path

import cv2
import numpy as np


class FlightUnavailable(RuntimeError):
    """Raised when the drone SDK or camera is not available in this environment."""


class FlightAdapter(abc.ABC):
    """Minimal flight interface used by the mission script."""

    def connect(self) -> None:
        raise NotImplementedError

    def takeoff(self, height_m: float) -> None:
        raise NotImplementedError

    def goto(self, x_m: float, y_m: float, z_m: float) -> None:
        raise NotImplementedError

    def capture_frame(self) -> np.ndarray | None:
        raise NotImplementedError

    def land(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class SimulatedAdapter(FlightAdapter):
    """Offline adapter: records commands and reads frames from disk."""

    def __init__(self, frame_path: str | None = None) -> None:
        self.frame_path = Path(frame_path) if frame_path is not None else None
        self.log: list[str] = []

    def connect(self) -> None:
        self.log.append("connect()")

    def takeoff(self, height_m: float) -> None:
        self.log.append(f"takeoff(height_m={height_m})")

    def goto(self, x_m: float, y_m: float, z_m: float) -> None:
        self.log.append(f"goto(x_m={x_m}, y_m={y_m}, z_m={z_m})")

    def capture_frame(self) -> np.ndarray | None:
        self.log.append("capture_frame()")
        if self.frame_path is None or not self.frame_path.exists():
            return None
        return cv2.imread(str(self.frame_path))

    def land(self) -> None:
        self.log.append("land()")

    def close(self) -> None:
        self.log.append("close()")


class PioneerAdapter(FlightAdapter):
    """Adapter for Pioneer Mini 2 via Pioneer SDK 2.

    The SDK is imported only in :meth:`connect` so importing this module (and
    therefore the vision pipeline) works on a laptop without the SDK.
    """

    def __init__(self) -> None:
        self._sdk = None
        self._drone = None

    def connect(self) -> None:
        try:
            import pioneer_sdk  # noqa: PLC0415 - lazy import keeps hardware optional
        except ImportError as exc:
            raise FlightUnavailable(
                "pioneer_sdk is not installed; run on the drone or use SimulatedAdapter"
            ) from exc
        self._sdk = pioneer_sdk
        # TODO: verify against Pioneer SDK 2 docs
        self._drone = pioneer_sdk.Pioneer()

    def takeoff(self, height_m: float) -> None:
        drone = self._require_drone()
        # TODO: verify against Pioneer SDK 2 docs (height may need a separate
        # altitude command before/after takeoff).
        drone.takeoff()

    def goto(self, x_m: float, y_m: float, z_m: float) -> None:
        drone = self._require_drone()
        # TODO: verify against Pioneer SDK 2 docs
        drone.go_to_local_point(x=x_m, y=y_m, z=z_m, yaw=0)

    def capture_frame(self) -> np.ndarray | None:
        # TODO: verify against Pioneer SDK 2 docs (camera frame/stream API).
        raise FlightUnavailable(
            "camera capture is not verified for PioneerAdapter yet; "
            "use SimulatedAdapter with saved frames"
        )

    def land(self) -> None:
        drone = self._require_drone()
        # TODO: verify against Pioneer SDK 2 docs
        drone.land()

    def close(self) -> None:
        drone = self._drone
        self._drone = None
        self._sdk = None
        if drone is not None:
            # TODO: verify against Pioneer SDK 2 docs
            drone.close()

    def _require_drone(self):
        if self._drone is None:
            raise FlightUnavailable("call connect() before sending flight commands")
        return self._drone
