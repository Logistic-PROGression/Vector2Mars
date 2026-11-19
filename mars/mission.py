"""Automated photo mission on top of :class:`mars.drone.DroneLink`.

Sequence: tilt the camera straight down, take off, climb to a target height
above the takeoff point, then fly the route to the target point in equal
steps. At every step the drone settles and takes a photo, so a frame where
the field is cut by the border can be replaced by an earlier one. After the
last stop it descends and lands at the target point.

The module has no curses and no CLI. It is one synchronous function so the
whole sequence can be exercised against ``SimulatedLink`` or a fake link with
a virtual clock (see ``tests/test_mission.py``). All steps, progress and
errors go to :class:`mars.flightlog.FlightLogger`.
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from mars.drone import DroneLink, DroneUnavailable, ServoScale, Telemetry
from mars.flightlog import FlightLogger

LOOP_S = 0.1
TELEMETRY_LOG_S = 1.0
TILT_HOLD_S = 1.0
# Proportional gain of the altitude controller, 1/s. With the capped climb
# speed the error decays over roughly a second, which is gentle for a small
# airframe.
ALTITUDE_GAIN = 0.8
# Proportional gain of the horizontal position controller, 1/s. The speed is
# capped by ``move_speed_ms``, so the drone slows down near the target instead
# of overshooting it.
HORIZONTAL_GAIN = 0.8


class MissionError(RuntimeError):
    """Raised when a mission step cannot be completed safely."""


@dataclass(frozen=True)
class MissionConfig:
    """Knobs of one photo mission.

    Every field has a CLI counterpart in ``photo_mission.py``; the defaults
    below are the source of truth for ``--help`` and for the tests.
    """

    altitude_m: float = 2.7
    max_altitude_m: float = 3.0
    target_x_m: float = 0.5
    target_y_m: float = 0.5
    tilt_deg: float = -90.0
    servo_zero_deg: float = 30.0
    servo_down_deg: float = -80.0
    servo_priority: str = "high"
    climb_speed_ms: float = 0.35
    descend_speed_ms: float = 0.4
    move_speed_ms: float = 0.4
    move_tolerance_m: float = 0.08
    tolerance_m: float = 0.05
    settle_s: float = 1.5
    tilt_settle_s: float = 1.0
    landing_height_m: float = 0.25
    # Number of stops on the route from the takeoff point to the target.
    # Stops are spread evenly from 0 to 100%: five stops give 0, 25, 50, 75
    # and 100% of the way, so even when the last frame is partly out of view
    # the earlier ones keep the whole field. One stop keeps the old behaviour
    # of a single photo at the target point.
    stop_count: int = 5
    photo_retries: int = 3
    photo_delay_s: float = 0.7
    climb_timeout_s: float = 60.0
    descend_timeout_s: float = 60.0
    move_timeout_s: float = 60.0

    def __post_init__(self) -> None:
        if self.altitude_m <= 0:
            raise ValueError("высота миссии должна быть больше 0")
        if self.max_altitude_m < self.altitude_m:
            raise ValueError(
                f"высота {self.altitude_m:.2f} м превышает защиту "
                f"--max-altitude {self.max_altitude_m:.2f} м"
            )
        if not -90.0 <= self.tilt_deg <= 0.0:
            raise ValueError("наклон камеры должен быть в диапазоне -90…0°")
        if not (math.isfinite(self.target_x_m) and math.isfinite(self.target_y_m)):
            raise ValueError("координаты цели должны быть числами")
        if self.climb_speed_ms <= 0 or self.descend_speed_ms <= 0:
            raise ValueError("скорости набора и снижения должны быть больше 0")
        if self.move_speed_ms <= 0:
            raise ValueError("горизонтальная скорость должна быть больше 0")
        if self.tolerance_m <= 0 or self.move_tolerance_m <= 0:
            raise ValueError("допуски по высоте и горизонтали должны быть больше 0")
        if not 0.0 <= self.landing_height_m < self.altitude_m:
            raise ValueError("высота начала посадки должна быть меньше высоты миссии")
        if self.settle_s < 0 or self.tilt_settle_s < 0:
            raise ValueError("паузы не могут быть отрицательными")
        if self.photo_retries < 1:
            raise ValueError("нужна хотя бы одна попытка фото")
        if self.stop_count < 1:
            raise ValueError("нужна хотя бы одна остановка для фото")
        if (
            self.climb_timeout_s <= 0
            or self.descend_timeout_s <= 0
            or self.move_timeout_s <= 0
        ):
            raise ValueError("таймауты должны быть больше 0")
        if self.servo_priority not in ("low", "medium", "high"):
            raise ValueError("приоритет сервопривода: low, medium или high")

    @property
    def stop_fractions(self) -> tuple[float, ...]:
        """Fractions of the route where the drone stops for a photo.

        ``0.0`` is the takeoff point, ``1.0`` is the target point. A single
        stop stays at the target so ``stop_count=1`` reproduces the previous
        one-photo mission.
        """
        if self.stop_count == 1:
            return (1.0,)
        return tuple(index / (self.stop_count - 1) for index in range(self.stop_count))


@dataclass
class MissionResult:
    ok: bool = False
    photo: Path | None = None
    photos: list[Path] = dataclasses.field(default_factory=list)
    photo_attempts: int = 0
    climbed_m: float | None = None
    elapsed_s: float = 0.0
    reason: str = ""
    interrupted: bool = False


def height_above_ground(telemetry: Telemetry, baseline: Telemetry) -> float | None:
    """Return the height gained since ``baseline``, or ``None`` without data.

    LPS z is preferred because it is metric and relative to the takeoff zone;
    the barometric ``altitude_m`` is the fallback when LPS is missing. Both
    sources are used as differences so an absolute barometric zero does not
    matter.
    """
    if telemetry.position is not None and baseline.position is not None:
        return float(telemetry.position[2]) - float(baseline.position[2])
    if telemetry.altitude_m is not None and baseline.altitude_m is not None:
        return float(telemetry.altitude_m) - float(baseline.altitude_m)
    return None


class _TiltHold:
    """Moves the servo down and keeps re-sending the angle.

    The board may re-stabilise the gimbal and silently undo a single manual
    command, so the angle is repeated once a second while the mission runs.
    """

    def __init__(self, link: DroneLink, config: MissionConfig, logger: FlightLogger) -> None:
        self.link = link
        self.config = config
        self.logger = logger
        self.scale = ServoScale(
            sdk_min=config.servo_down_deg, sdk_max=config.servo_zero_deg
        )
        self.target = self.scale.clamp(config.tilt_deg)
        self.sdk_angle = self.scale.to_sdk(self.target)
        self.active = False
        self._last_send = 0.0

    def command(self, clock: Callable[[], float]) -> bool:
        """Send the tilt once; return False when the servo is unavailable."""
        try:
            ok = self.link.set_camera_angle(
                self.sdk_angle, priority=self.config.servo_priority, attempts=3
            )
        except DroneUnavailable as exc:
            self.logger.log(
                "tilt_unavailable",
                error=str(exc),
                physical_deg=round(self.target, 1),
                sdk_deg=round(self.sdk_angle, 1),
            )
            return False
        except Exception as exc:  # pragma: no cover - hardware specific
            self.logger.log(
                "tilt_failed",
                error=str(exc),
                physical_deg=round(self.target, 1),
                sdk_deg=round(self.sdk_angle, 1),
            )
            return False
        self.active = bool(ok)
        self._last_send = clock()
        self.logger.log(
            "tilt",
            ok=bool(ok),
            physical_deg=round(self.target, 1),
            sdk_deg=round(self.sdk_angle, 1),
            priority=self.config.servo_priority,
        )
        return bool(ok)

    def hold(self, clock: Callable[[], float]) -> None:
        if not self.active:
            return
        if clock() - self._last_send < TILT_HOLD_S:
            return
        self._last_send = clock()
        try:
            self.link.set_camera_angle(
                self.sdk_angle, priority=self.config.servo_priority, attempts=1
            )
        except Exception:
            pass


def _safe_telemetry(link: DroneLink, logger: FlightLogger) -> Telemetry:
    try:
        return link.telemetry()
    except Exception as exc:
        logger.log("telemetry_error", error=str(exc))
        return Telemetry()


def _read_baseline(
    link: DroneLink,
    logger: FlightLogger,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> Telemetry:
    """Wait briefly for a usable altitude reference before takeoff."""
    last = Telemetry()
    for _ in range(5):
        last = _safe_telemetry(link, logger)
        if last.position is not None or last.altitude_m is not None:
            return last
        sleep(0.2)
    raise MissionError("нет телеметрии высоты (LPS/альтиметр): миссия отменена")


def _read_takeoff_position(
    link: DroneLink,
    logger: FlightLogger,
    sleep: Callable[[float], None],
) -> tuple[float, float, float]:
    """Return the LPS point where the drone took off.

    The LPS origin may be reset during takeoff, so the first samples can still
    belong to the old frame. Reading several times and keeping the last one
    gives the coordinates the mission should treat as "from the start".
    """
    point: tuple[float, float, float] | None = None
    for _ in range(5):
        telemetry = _safe_telemetry(link, logger)
        if telemetry.position is not None:
            point = (
                float(telemetry.position[0]),
                float(telemetry.position[1]),
                float(telemetry.position[2]),
            )
            sleep(0.2)

    if point is None:
        raise MissionError("нет LPS-координат после взлёта: миссия отменена")

    logger.log(
        "takeoff_reference",
        x=round(point[0], 2),
        y=round(point[1], 2),
        z=round(point[2], 2),
    )
    return point


def _climb(
    link: DroneLink,
    config: MissionConfig,
    baseline: Telemetry,
    logger: FlightLogger,
    notice: Callable[[str], None],
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    tilt: _TiltHold,
) -> float:
    started = clock()
    last_log = started
    height: float | None = None
    while True:
        telemetry = _safe_telemetry(link, logger)
        height = height_above_ground(telemetry, baseline)
        if height is None:
            raise MissionError("нет данных о высоте: набор высоты остановлен")

        error = config.altitude_m - height
        if abs(error) <= config.tolerance_m:
            link.set_velocity(0.0, 0.0, 0.0, 0.0, interval=0.2)
            elapsed = clock() - started
            logger.log(
                "climb_done",
                height_m=round(height, 2),
                target_m=config.altitude_m,
                elapsed_s=round(elapsed, 1),
            )
            notice(f"Высота достигнута: {height:.2f} м за {elapsed:.1f} с")
            return height

        if clock() - started > config.climb_timeout_s:
            raise MissionError(
                f"высота {config.altitude_m:.2f} м не достигнута за "
                f"{config.climb_timeout_s:.0f} с (сейчас {height:.2f} м)"
            )

        vz = max(-config.climb_speed_ms, min(config.climb_speed_ms, error * ALTITUDE_GAIN))
        link.set_velocity(0.0, 0.0, vz, 0.0, interval=0.2)
        tilt.hold(clock)

        if clock() - last_log >= TELEMETRY_LOG_S:
            last_log = clock()
            logger.log(
                "climb_progress",
                height_m=round(height, 2),
                target_m=config.altitude_m,
                vz=round(vz, 2),
            )
        sleep(LOOP_S)


def _move_to_point(
    link: DroneLink,
    config: MissionConfig,
    baseline: Telemetry,
    takeoff_point: tuple[float, float, float],
    offset_x_m: float,
    offset_y_m: float,
    stop: int,
    logger: FlightLogger,
    notice: Callable[[str], None],
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    tilt: _TiltHold,
) -> None:
    """Fly horizontally to a point relative to the takeoff point.

    The target is expressed in the LPS frame, while ``set_velocity`` is
    body-fixed, so the world-frame velocity is rotated by the current yaw. The
    altitude is held with the same proportional controller as the climb. The
    same function serves every stop on the route, only the offset changes.
    """
    target_x = takeoff_point[0] + offset_x_m
    target_y = takeoff_point[1] + offset_y_m
    started = clock()
    last_log = started
    notice(
        f"Остановка {stop + 1}/{config.stop_count}: "
        f"полёт к x={offset_x_m:.2f}, y={offset_y_m:.2f} м от взлёта…"
    )

    while True:
        telemetry = _safe_telemetry(link, logger)
        if telemetry.position is None:
            raise MissionError("нет LPS-координат: горизонтальный полёт остановлен")

        dx = target_x - float(telemetry.position[0])
        dy = target_y - float(telemetry.position[1])
        distance = math.hypot(dx, dy)

        if distance <= config.move_tolerance_m:
            link.set_velocity(0.0, 0.0, 0.0, 0.0, interval=0.2)
            elapsed = clock() - started
            logger.log(
                "move_done",
                stop=stop,
                x=round(float(telemetry.position[0]), 2),
                y=round(float(telemetry.position[1]), 2),
                target_x=round(target_x, 2),
                target_y=round(target_y, 2),
                error_m=round(distance, 2),
                elapsed_s=round(elapsed, 1),
            )
            notice(
                f"Точка достигнута: x={telemetry.position[0]:.2f}, "
                f"y={telemetry.position[1]:.2f} м за {elapsed:.1f} с"
            )
            return

        if clock() - started > config.move_timeout_s:
            raise MissionError(
                f"точка x={target_x:.2f}, y={target_y:.2f} м не достигнута за "
                f"{config.move_timeout_s:.0f} с (ошибка {distance:.2f} м)"
            )

        height = height_above_ground(telemetry, baseline)
        if height is None:
            raise MissionError("нет данных о высоте: горизонтальный полёт остановлен")
        altitude_error = config.altitude_m - height
        vertical_limit = (
            config.climb_speed_ms if altitude_error >= 0 else config.descend_speed_ms
        )
        vz = max(
            -vertical_limit,
            min(vertical_limit, altitude_error * ALTITUDE_GAIN),
        )

        speed = min(config.move_speed_ms, distance * HORIZONTAL_GAIN)
        world_vx = speed * dx / distance
        world_vy = speed * dy / distance
        yaw = math.radians(telemetry.yaw_deg or 0.0)
        body_vx = world_vx * math.cos(yaw) + world_vy * math.sin(yaw)
        body_vy = -world_vx * math.sin(yaw) + world_vy * math.cos(yaw)
        link.set_velocity(body_vx, body_vy, vz, 0.0, interval=0.2)
        tilt.hold(clock)

        if clock() - last_log >= TELEMETRY_LOG_S:
            last_log = clock()
            logger.log(
                "move_progress",
                stop=stop,
                x=round(float(telemetry.position[0]), 2),
                y=round(float(telemetry.position[1]), 2),
                error_m=round(distance, 2),
                vx=round(body_vx, 2),
                vy=round(body_vy, 2),
                height_m=round(height, 2),
            )
        sleep(LOOP_S)


def _settle(
    link: DroneLink,
    config: MissionConfig,
    tilt: _TiltHold,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> None:
    """Hover in place after the climb so the frame and the photo are steady."""
    deadline = clock() + config.settle_s
    while clock() < deadline:
        link.set_velocity(0.0, 0.0, 0.0, 0.0, interval=0.2)
        tilt.hold(clock)
        sleep(LOOP_S)
    link.set_velocity(0.0, 0.0, 0.0, 0.0, interval=0.2)


def _take_photo(
    link: DroneLink,
    config: MissionConfig,
    logger: FlightLogger,
    photo_dir: str | Path,
    notice: Callable[[str], None],
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    tilt: _TiltHold,
    stop: int | None = None,
) -> tuple[Path | None, int]:
    for attempt in range(1, config.photo_retries + 1):
        path = None
        try:
            path = link.take_photo(photo_dir)
        except Exception as exc:
            logger.log("photo_error", attempt=attempt, stop=stop, error=str(exc))
        if path is not None:
            logger.log("photo", path=str(path), attempt=attempt, stop=stop)
            notice(f"Фото сохранено: {path}")
            return Path(path), attempt
        logger.log(
            "photo_failed",
            attempt=attempt,
            stop=stop,
            camera_error=getattr(link, "camera_error", None),
        )
        notice(f"Кадр не получен (попытка {attempt} из {config.photo_retries})")
        if attempt < config.photo_retries:
            sleep(config.photo_delay_s)
            tilt.hold(clock)
    return None, config.photo_retries


def _descend(
    link: DroneLink,
    config: MissionConfig,
    baseline: Telemetry,
    logger: FlightLogger,
    notice: Callable[[str], None],
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    tilt: _TiltHold,
) -> None:
    started = clock()
    last_log = started
    while True:
        telemetry = _safe_telemetry(link, logger)
        height = height_above_ground(telemetry, baseline)
        if height is None:
            raise MissionError("нет данных о высоте: снижение остановлено")

        if height <= config.landing_height_m + config.tolerance_m:
            link.set_velocity(0.0, 0.0, 0.0, 0.0, interval=0.2)
            elapsed = clock() - started
            logger.log(
                "descent_done",
                height_m=round(height, 2),
                elapsed_s=round(elapsed, 1),
            )
            # The landing phase is reported in the JSONL log only: console
            # output stops at the last stop of the mission.
            return

        if clock() - started > config.descend_timeout_s:
            raise MissionError(
                f"снижение не завершено за {config.descend_timeout_s:.0f} с "
                f"(сейчас {height:.2f} м)"
            )

        error = height - config.landing_height_m
        vz = -min(config.descend_speed_ms, error * ALTITUDE_GAIN)
        link.set_velocity(0.0, 0.0, vz, 0.0, interval=0.2)
        tilt.hold(clock)

        if clock() - last_log >= TELEMETRY_LOG_S:
            last_log = clock()
            logger.log(
                "descent_progress",
                height_m=round(height, 2),
                vz=round(vz, 2),
            )
        sleep(LOOP_S)


def _land(link: DroneLink, logger: FlightLogger, notice: Callable[[str], None]) -> None:
    """Land through the board; the console stays silent about the landing."""
    logger.log("land_requested")
    try:
        ok = link.land()
    except Exception as exc:
        logger.log("land_error", error=str(exc))
        raise MissionError(f"посадка не выполнена: {exc}") from exc
    logger.log("land", ok=bool(ok))
    if not ok:
        raise MissionError("посадка не подтверждена бортом")


def _safe_land(
    link: DroneLink, logger: FlightLogger, notice: Callable[[str], None]
) -> None:
    """Best-effort landing for error and interrupt paths; never raises.

    The outcome goes to the log only; the console says nothing about it.
    """
    try:
        link.stop()
    except Exception:
        pass
    try:
        ok = link.land()
    except Exception as exc:
        logger.log("auto_land_failed", error=str(exc))
        return
    logger.log("auto_land", ok=bool(ok))


def run_photo_mission(
    link: DroneLink,
    config: MissionConfig,
    logger: FlightLogger,
    photo_dir: str | Path = "photos",
    *,
    notice: Callable[[str], None] = print,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> MissionResult:
    """Run the full sequence and always try to put the drone back on the ground.

    ``clock`` and ``sleep`` are injectable so tests can drive the mission with
    a virtual timeline instead of real seconds.
    """
    started = clock()
    result = MissionResult()
    tilt = _TiltHold(link, config, logger)
    logger.log(
        "mission_start",
        **dataclasses.asdict(config),
        photo_dir=str(photo_dir),
    )
    notice(
        f"Миссия: камера {config.tilt_deg:.0f}°, высота {config.altitude_m:.2f} м, "
        f"точка ({config.target_x_m:.2f}, {config.target_y_m:.2f}) м от взлёта, "
        f"остановок {config.stop_count}, фото, посадка"
    )

    airborne = False
    try:
        baseline = _read_baseline(link, logger, clock, sleep)
        logger.log("ground_reference", **baseline.as_dict())

        # 1. Point the camera down before the climb.
        if tilt.command(clock):
            notice(f"Камера: {tilt.target:.0f}° (SDK {tilt.sdk_angle:.0f}°)")
            sleep(config.tilt_settle_s)
        else:
            notice(
                "Наклон камеры недоступен (нужен запуск на борту с --local); "
                "продолжаю без наклона"
            )

        # 2. Take off.
        notice("Взлёт…")
        logger.log("takeoff_requested")
        try:
            ok = link.takeoff()
        except Exception as exc:
            logger.log("takeoff_error", error=str(exc))
            raise MissionError(f"взлёт не выполнен: {exc}") from exc
        logger.log("takeoff", ok=bool(ok))
        if not ok:
            raise MissionError("взлёт не подтверждён бортом")
        airborne = True
        notice("Взлёт выполнен")
        takeoff_point = _read_takeoff_position(link, logger, sleep)

        # 3. Climb to the requested height above the takeoff point.
        notice(f"Набор высоты до {config.altitude_m:.2f} м…")
        result.climbed_m = _climb(
            link, config, baseline, logger, notice, clock, sleep, tilt
        )

        # 4. Fly the route in equal steps, stopping to photograph at each one.
        for stop, fraction in enumerate(config.stop_fractions):
            logger.log("stop_start", stop=stop, fraction=fraction)
            if fraction > 0.0:
                _move_to_point(
                    link,
                    config,
                    baseline,
                    takeoff_point,
                    config.target_x_m * fraction,
                    config.target_y_m * fraction,
                    stop,
                    logger,
                    notice,
                    clock,
                    sleep,
                    tilt,
                )

            # 5. Wait for the airframe and the frame to settle.
            _settle(link, config, tilt, clock, sleep)

            # 6. Photo at this stop.
            photo, attempts = _take_photo(
                link, config, logger, photo_dir, notice, clock, sleep, tilt, stop=stop
            )
            result.photo_attempts += attempts
            if photo is not None:
                result.photos.append(photo)
                result.photo = photo
            logger.log(
                "stop_done",
                stop=stop,
                fraction=fraction,
                photo=None if photo is None else str(photo),
            )

        if not result.photos:
            result.reason = "фото не получено"

        # 7. Descend under control, then land.
        _descend(link, config, baseline, logger, notice, clock, sleep, tilt)
        _land(link, logger, notice)
        airborne = False

        result.ok = bool(result.photos)
        if result.ok:
            notice(f"Миссия завершена: кадров {len(result.photos)}")
        else:
            notice("Миссия завершена без фото")
    except KeyboardInterrupt:
        result.interrupted = True
        result.reason = "прервано оператором"
        logger.log("mission_interrupted")
        notice("Прервано")
    except MissionError as exc:
        result.reason = str(exc)
        logger.log("mission_error", error=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        result.reason = str(exc)
        logger.log("mission_error", error=str(exc), error_type=type(exc).__name__)
    finally:
        if airborne:
            _safe_land(link, logger, notice)
        else:
            try:
                link.stop()
            except Exception:
                pass
        result.elapsed_s = round(clock() - started, 1)
        logger.log(
            "mission_complete",
            ok=result.ok,
            reason=result.reason,
            photo=None if result.photo is None else str(result.photo),
            photos=[str(photo) for photo in result.photos],
            photo_attempts=result.photo_attempts,
            climbed_m=None if result.climbed_m is None else round(result.climbed_m, 2),
            elapsed_s=result.elapsed_s,
            interrupted=result.interrupted,
        )
    return result
