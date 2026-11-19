"""Thin flight adapter so vision code never imports hardware SDKs.

``mars.vision`` only consumes frames. All Pioneer SDK 2 usage is confined to
:class:`PioneerAdapter`, which delegates to :class:`mars.drone.DroneLink` and
therefore imports the SDK lazily; the vision pipeline still runs and is tested
offline.
"""

from __future__ import annotations

import abc
from pathlib import Path

import cv2
import numpy as np

from mars.drone import DEFAULT_TCP, DroneLink, DroneUnavailable


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
    """Adapter for Pioneer Mini 2 via :class:`mars.drone.DroneLink`."""

    def __init__(self, tcp: str = DEFAULT_TCP, local: bool = False) -> None:
        self.tcp = tcp
        self.local = local
        self._link: DroneLink | None = None

    def connect(self) -> None:
        link = DroneLink(tcp=self.tcp, local=self.local)
        try:
            link.connect()
        except DroneUnavailable as exc:
            raise FlightUnavailable(str(exc)) from exc
        self._link = link

    def takeoff(self, height_m: float) -> None:
        # SDK 2 takeoff() climbs to its configured altitude; height_m is kept
        # for the mission interface and applied by the caller via goto().
        self._require().takeoff()

    def goto(self, x_m: float, y_m: float, z_m: float) -> None:
        self._require().goto_local(x_m, y_m, z_m)

    def capture_frame(self) -> np.ndarray | None:
        return self._require().grab_frame()

    def land(self) -> None:
        self._require().land()

    def close(self) -> None:
        link, self._link = self._link, None
        if link is not None:
            link.close()

    def _require(self) -> DroneLink:
        if self._link is None:
            raise FlightUnavailable("call connect() before sending flight commands")
        return self._link
