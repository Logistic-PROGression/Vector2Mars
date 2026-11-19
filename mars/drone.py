"""Pioneer Mini 2 teleoperation link built on Pioneer SDK 2.

The TUI talks to this module only. The SDK is imported inside
:meth:`DroneLink.connect`, so importing the module (and running the unit tests)
works on a machine without the SDK or the drone.

Verified against ``pioneer_sdk2`` 0.11.0 sources:

* ``Pioneer(tcp="host:port", logger=False)``, ``arm()``, ``takeoff()``,
  ``land()``, ``disarm()``, ``close_connection()``
* ``set_manual_speed_body_fixed(vx, vy, vz, yaw_rate, interval)`` with
  ``yaw_rate`` in rad/s and velocities in m/s
* ``get_fly_state()``, ``get_battery_status()``, ``get_local_position_lps()``,
  ``get_local_yaw_lps()``, ``get_altitude()``
* ``Camera(camera_type=CameraType.MAIN, camera_ip="10.42.0.1:8554")`` for the
  remote RTSP stream, ``get_cv_frame(timeout)``
* ``ServoCamera().set_angle(deg)``, range -80..30, Unix socket only
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

DEFAULT_TCP = "10.42.0.1:20556"
DEFAULT_CAMERA = "10.42.0.1:8554"
LOCAL_TCP = "127.0.0.1:20556"
SERVO_MIN_ANGLE = -80.0
SERVO_MAX_ANGLE = 30.0


SERVO_PHYSICAL_MIN = -90.0
SERVO_PHYSICAL_MAX = 0.0
SERVO_SDK_AT_DOWN = -80.0
SERVO_SDK_AT_HORIZON = 30.0


@dataclass(frozen=True)
class ServoScale:
    """Maps physical camera pitch to the SDK servo angle.

    The SDK range (-80..30) is not the optical range: on the Mini 2 the
    camera looks horizontally at SDK +30 and straight down at SDK -80.
    Physical angles therefore run from -90 (down) to 0 (horizon), and the
    mapping is linear between the two measured endpoints.
    """

    physical_min: float = SERVO_PHYSICAL_MIN
    physical_max: float = SERVO_PHYSICAL_MAX
    sdk_min: float = SERVO_SDK_AT_DOWN
    sdk_max: float = SERVO_SDK_AT_HORIZON

    def clamp(self, physical: float) -> float:
        return max(self.physical_min, min(self.physical_max, float(physical)))

    def to_sdk(self, physical: float) -> float:
        value = self.clamp(physical)
        span = self.physical_max - self.physical_min
        if span == 0:
            return self.sdk_max
        ratio = (value - self.physical_min) / span
        return self.sdk_min + ratio * (self.sdk_max - self.sdk_min)

    def to_physical(self, sdk: float) -> float:
        span = self.sdk_max - self.sdk_min
        if span == 0:
            return self.physical_max
        ratio = (float(sdk) - self.sdk_min) / span
        return self.physical_min + ratio * (self.physical_max - self.physical_min)


class DroneUnavailable(RuntimeError):
    """Raised when the SDK, the drone or the camera cannot be reached."""


@dataclass(frozen=True)
class Telemetry:
    """Snapshot rendered by the TUI and written to the flight log."""

    state: str = "UNKNOWN"
    battery_v: float | None = None
    battery_c: float | None = None
    position: tuple[float, float, float] | None = None
    altitude_m: float | None = None
    yaw_deg: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "battery_v": None if self.battery_v is None else round(self.battery_v, 2),
            "battery_c": None if self.battery_c is None else round(self.battery_c, 1),
            "x": None if self.position is None else round(self.position[0], 2),
            "y": None if self.position is None else round(self.position[1], 2),
            "z": None if self.position is None else round(self.position[2], 2),
            "altitude_m": None if self.altitude_m is None else round(self.altitude_m, 2),
            "yaw_deg": None if self.yaw_deg is None else round(self.yaw_deg, 1),
        }


class DroneLink:
    """Thin, defensive wrapper around the Pioneer SDK 2 classes."""

    def __init__(
        self,
        tcp: str = DEFAULT_TCP,
        camera_ip: str = DEFAULT_CAMERA,
        local: bool = False,
        with_camera: bool = True,
    ) -> None:
        self.tcp = tcp
        self.camera_ip = camera_ip
        self.local = local
        self.with_camera = with_camera
        self._sdk = None
        self._drone = None
        self._camera = None
        self._servo = None
        self._camera_error: str | None = None

    # -- lifecycle ---------------------------------------------------------

    def connect(self) -> None:
        try:
            import pioneer_sdk2  # noqa: PLC0415 - hardware import stays lazy
        except ImportError as exc:
            raise DroneUnavailable(
                "pioneer_sdk2 не установлен. На ноутбуке: "
                "python3 -m pip install pioneer-sdk2"
            ) from exc

        address = LOCAL_TCP if self.local else self.tcp
        try:
            # The SDK calls exit() on a failed TCP connect, hence SystemExit.
            self._drone = pioneer_sdk2.Pioneer(tcp=address, logger=False)
        except SystemExit as exc:
            raise DroneUnavailable(
                f"не удалось подключиться к дрону по {address}. "
                "Проверьте Wi-Fi дрона и маршрут: ip route get 10.42.0.1"
            ) from exc
        except Exception as exc:  # pragma: no cover - depends on hardware
            raise DroneUnavailable(f"ошибка подключения к дрону: {exc}") from exc

        self._sdk = pioneer_sdk2
        if self.with_camera:
            self._open_camera(pioneer_sdk2)

    def _open_camera(self, sdk) -> None:
        # On board the SDK uses the local GStreamer driver, whose Camera()
        # accepts no camera_ip; the laptop wheel uses RTSP and needs it.
        driver = None
        try:
            driver = sdk.CONFIG["cameras"]["driver"]
        except Exception:
            pass
        use_local_driver = self.local or driver == "gstreamer"
        try:
            if use_local_driver:
                self._camera = sdk.Camera(camera_type=sdk.CameraType.MAIN)
            else:
                try:
                    self._camera = sdk.Camera(
                        camera_type=sdk.CameraType.MAIN, camera_ip=self.camera_ip
                    )
                except TypeError:  # board build without camera_ip support
                    self._camera = sdk.Camera(camera_type=sdk.CameraType.MAIN)
            self._camera_error = None
        except Exception as exc:  # camera is optional: teleop still works
            self._camera = None
            self._camera_error = str(exc)

    def close(self) -> None:
        self.stop()
        if self._camera is not None:
            try:
                self._camera.stop()
            except Exception:
                pass
            self._camera = None
        drone, self._drone = self._drone, None
        if drone is not None:
            try:
                drone.close_connection()
            except Exception:
                pass
        self._sdk = None

    # -- flight ------------------------------------------------------------

    def arm(self) -> bool:
        return bool(self._require().arm(timeout=5, retries=1))

    def takeoff(self) -> bool:
        """Arm the motors and wait for the climb to finish."""
        drone = self._require()
        if not drone.arm(timeout=5, retries=1):
            return False
        return bool(drone.takeoff())

    def land(self) -> bool:
        return bool(self._require().land())

    def disarm(self) -> bool:
        return bool(self._require().disarm())

    def stop(self) -> None:
        """Send a zero velocity so the drone holds position."""
        if self._drone is None:
            return
        try:
            self._drone.set_manual_speed_body_fixed(0.0, 0.0, 0.0, 0.0, 0.2)
        except Exception:
            pass

    def set_velocity(
        self,
        vx: float,
        vy: float,
        vz: float,
        yaw_rate_deg: float,
        interval: float = 0.2,
    ) -> None:
        """Body-fixed velocity: x forward, y left, z up, yaw in deg/s."""
        if self._drone is None:
            raise DroneUnavailable("нет соединения с дроном")
        yaw_rate = math.radians(float(yaw_rate_deg))
        try:
            self._drone.set_manual_speed_body_fixed(
                float(vx), float(vy), float(vz), yaw_rate, float(interval)
            )
        except Exception as exc:
            raise DroneUnavailable(f"команда скорости отклонена: {exc}") from exc

    def goto_local(self, x: float, y: float, z: float, yaw: float = 0.0) -> None:
        self._require().go_to_local_point(x=float(x), y=float(y), z=float(z), yaw=float(yaw))

    # -- telemetry ---------------------------------------------------------

    def telemetry(self) -> Telemetry:
        drone = self._drone
        if drone is None:
            return Telemetry()

        state = "UNKNOWN"
        try:
            raw_state = drone.get_fly_state()
            state = getattr(raw_state, "name", str(raw_state))
        except Exception:
            pass

        battery_v = battery_c = None
        try:
            battery = drone.get_battery_status()
            if battery is not None:
                battery_v, battery_c = battery[0], battery[1]
        except Exception:
            pass

        position = altitude = yaw = None
        try:
            position = drone.get_local_position_lps()
        except Exception:
            pass
        try:
            altitude = drone.get_altitude()
        except Exception:
            pass
        try:
            yaw = drone.get_local_yaw_lps()
        except Exception:
            pass

        return Telemetry(
            state=state,
            battery_v=battery_v,
            battery_c=battery_c,
            position=position,
            altitude_m=altitude,
            yaw_deg=yaw,
        )

    # -- camera ------------------------------------------------------------

    @property
    def camera_error(self) -> str | None:
        return self._camera_error

    def grab_frame(self, timeout: float = 1.0) -> np.ndarray | None:
        if self._camera is None:
            return None
        try:
            frame = self._camera.get_cv_frame(timeout=timeout)
        except TypeError:  # some board builds expose get_cv_frame() only
            try:
                frame = self._camera.get_cv_frame()
            except Exception:
                return None
        except Exception:
            return None
        if frame is None or getattr(frame, "size", 0) == 0:
            return None
        return frame

    def take_photo(self, directory: str | Path = "photos") -> Path | None:
        frame = self.grab_frame()
        if frame is None:
            return None
        folder = Path(directory).expanduser()
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        path = folder / f"photo_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
        try:
            if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                return None
        except cv2.error:
            return None
        return path

    def set_camera_angle(
        self, angle: float, priority: str = "high", attempts: int = 3
    ) -> bool:
        """Tilt the camera servo. Works only on board (Unix socket).

        The SDK only confirms that the datagram reached ``/tmp/servo.sock``;
        there is no angle read-back, so ``True`` does not prove the camera
        moved. Manual tilt uses HIGH priority (LOW is easily overridden by
        the board) and repeats the command to survive a dropped datagram.
        """
        angle = max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, float(angle)))
        # The board daemon accepts whole degrees; a fractional payload such as
        # "17.77777777777778" is ignored, so round before sending.
        angle = int(round(angle))
        if self._servo is None:
            try:
                self._servo = self._sdk.ServoCamera()  # type: ignore[union-attr]
            except Exception as exc:
                raise DroneUnavailable(
                    "сервопривод камеры доступен только на борту дрона "
                    "(в удалённом режиме SDK не имеет к нему доступа)"
                ) from exc
        servo_priority = getattr(
            self._sdk.ServoPriority,
            str(priority).upper(),
            self._sdk.ServoPriority.HIGH,
        )
        sent = False
        for _ in range(max(1, int(attempts))):
            if self._servo.set_angle(angle, servo_priority):
                sent = True
            time.sleep(0.05)
        return sent

    # -- helpers -----------------------------------------------------------

    def _require(self):
        if self._drone is None:
            raise DroneUnavailable("нет соединения с дроном")
        return self._drone


class SimulatedLink(DroneLink):
    """Hardware-free stand-in used by ``--dry-run`` and the unit tests."""

    def __init__(self, frame_path: str | Path | None = "samples/image.jpg") -> None:
        super().__init__(local=True, with_camera=False)
        self.frame_path = Path(frame_path) if frame_path else None
        self.state = "ON_LAND"
        self.position = [0.0, 0.0, 0.0]
        self.yaw_deg = 0.0
        self.angle = 0.0
        self.photos: list[Path] = []
        self.velocity = (0.0, 0.0, 0.0, 0.0)
        self._last_update = time.monotonic()

    def connect(self) -> None:
        self._last_update = time.monotonic()

    def close(self) -> None:
        self.stop()

    def takeoff(self) -> bool:
        self.state = "IN_SKY"
        self.position[2] = 1.0
        return True

    def land(self) -> bool:
        self.state = "ON_LAND"
        self.position[2] = 0.0
        return True

    def arm(self) -> bool:
        self.state = "ARMED"
        return True

    def disarm(self) -> bool:
        self.state = "ON_LAND"
        return True

    def stop(self) -> None:
        self.velocity = (0.0, 0.0, 0.0, 0.0)

    def set_velocity(self, vx, vy, vz, yaw_rate_deg, interval=0.2) -> None:
        self._integrate()
        self.velocity = (float(vx), float(vy), float(vz), float(yaw_rate_deg))

    def _integrate(self) -> None:
        now = time.monotonic()
        dt = min(now - self._last_update, 0.5)
        self._last_update = now
        vx, vy, vz, yaw_rate = self.velocity
        self.position[0] += vx * dt
        self.position[1] += vy * dt
        self.position[2] = max(0.0, self.position[2] + vz * dt)
        self.yaw_deg = (self.yaw_deg + yaw_rate * dt + 180.0) % 360.0 - 180.0

    def telemetry(self) -> Telemetry:
        self._integrate()
        return Telemetry(
            state=self.state,
            battery_v=8.2,
            battery_c=31.0,
            position=tuple(self.position),
            altitude_m=self.position[2],
            yaw_deg=self.yaw_deg,
        )

    def grab_frame(self, timeout: float = 1.0) -> np.ndarray | None:
        if self.frame_path is not None and self.frame_path.exists():
            frame = cv2.imread(str(self.frame_path))
            if frame is not None:
                return frame
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            "SIMULATED CAMERA",
            (150, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    def take_photo(self, directory: str | Path = "photos") -> Path | None:
        frame = self.grab_frame()
        folder = Path(directory).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"photo_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
        cv2.imwrite(str(path), frame)
        self.photos.append(path)
        return path

    def set_camera_angle(
        self, angle: float, priority: str = "high", attempts: int = 3
    ) -> bool:
        self.angle = max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, float(angle)))
        return True
