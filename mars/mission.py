"""Automated photo mission on top of :class:`mars.drone.DroneLink`.

Sequence: tilt the camera straight down, take off, climb to a target height
above the takeoff point, settle, take a photo, descend and land.

The module has no curses and no CLI. It is one synchronous function so the
whole sequence can be exercised against ``SimulatedLink`` or a fake link with
a virtual clock (see ``tests/test_mission.py``). All steps, progress and
errors go to :class:`mars.flightlog.FlightLogger`.
"""

from __future__ import annotations

import dataclasses
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
    tilt_deg: float = -90.0
    servo_zero_deg: float = 30.0
    servo_down_deg: float = -80.0
    servo_priority: str = "high"
    climb_speed_ms: float = 0.35
    descend_speed_ms: float = 0.4
    tolerance_m: float = 0.05
    settle_s: float = 1.5
    tilt_settle_s: float = 1.0
    landing_height_m: float = 0.25
    photo_retries: int = 3
    photo_delay_s: float = 0.7
    climb_timeout_s: float = 60.0
    descend_timeout_s: float = 60.0

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
        if self.climb_speed_ms <= 0 or self.descend_speed_ms <= 0:
            raise ValueError("скорости набора и снижения должны быть больше 0")
        if self.tolerance_m <= 0:
            raise ValueError("допуск по высоте должен быть больше 0")
        if not 0.0 <= self.landing_height_m < self.altitude_m:
            raise ValueError("высота начала посадки должна быть меньше высоты миссии")
        if self.settle_s < 0 or self.tilt_settle_s < 0:
            raise ValueError("паузы не могут быть отрицательными")
        if self.photo_retries < 1:
            raise ValueError("нужна хотя бы одна попытка фото")
        if self.climb_timeout_s <= 0 or self.descend_timeout_s <= 0:
            raise ValueError("таймауты должны быть больше 0")
        if self.servo_priority not in ("low", "medium", "high"):
            raise ValueError("приоритет сервопривода: low, medium или high")


@dataclass
class MissionResult:
    ok: bool = False
    photo: Path | None = None
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
) -> tuple[Path | None, int]:
    for attempt in range(1, config.photo_retries + 1):
        path = None
        try:
            path = link.take_photo(photo_dir)
        except Exception as exc:
            logger.log("photo_error", attempt=attempt, error=str(exc))
        if path is not None:
            logger.log("photo", path=str(path), attempt=attempt)
            notice(f"Фото сохранено: {path}")
            return Path(path), attempt
        logger.log(
            "photo_failed",
            attempt=attempt,
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
            notice(f"Снижение: {height:.2f} м, посадка")
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
    logger.log("land_requested")
    try:
        ok = link.land()
    except Exception as exc:
        logger.log("land_error", error=str(exc))
        raise MissionError(f"посадка не выполнена: {exc}") from exc
    logger.log("land", ok=bool(ok))
    if not ok:
        raise MissionError("посадка не подтверждена бортом")
    notice("Посадка выполнена")


def _safe_land(
    link: DroneLink, logger: FlightLogger, notice: Callable[[str], None]
) -> None:
    """Best-effort landing for error and interrupt paths; never raises."""
    try:
        link.stop()
    except Exception:
        pass
    try:
        ok = link.land()
    except Exception as exc:
        logger.log("auto_land_failed", error=str(exc))
        notice(f"Аварийная посадка не подтверждена: {exc}")
        return
    logger.log("auto_land", ok=bool(ok))
    notice("Посадка выполнена" if ok else "Аварийная посадка не подтверждена")


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
        "фото, посадка"
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

        # 3. Climb to the requested height above the takeoff point.
        notice(f"Набор высоты до {config.altitude_m:.2f} м…")
        result.climbed_m = _climb(
            link, config, baseline, logger, notice, clock, sleep, tilt
        )

        # 4. Wait for the airframe and the frame to settle.
        _settle(link, config, tilt, clock, sleep)

        # 5. Photo.
        result.photo, result.photo_attempts = _take_photo(
            link, config, logger, photo_dir, notice, clock, sleep, tilt
        )
        if result.photo is None:
            result.reason = "фото не получено"

        # 6. Descend under control, then land.
        _descend(link, config, baseline, logger, notice, clock, sleep, tilt)
        _land(link, logger, notice)
        airborne = False

        result.ok = result.photo is not None
        if result.ok:
            notice(f"Миссия завершена: {result.photo}")
        else:
            notice("Миссия завершена без фото")
    except KeyboardInterrupt:
        result.interrupted = True
        result.reason = "прервано оператором"
        logger.log("mission_interrupted")
        notice("Прервано: аварийная посадка")
    except MissionError as exc:
        result.reason = str(exc)
        logger.log("mission_error", error=str(exc))
        notice(f"Ошибка миссии: {exc}")
    except Exception as exc:  # pragma: no cover - defensive
        result.reason = str(exc)
        logger.log("mission_error", error=str(exc), error_type=type(exc).__name__)
        notice(f"Ошибка миссии: {exc}")
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
            photo_attempts=result.photo_attempts,
            climbed_m=None if result.climbed_m is None else round(result.climbed_m, 2),
            elapsed_s=result.elapsed_s,
            interrupted=result.interrupted,
        )
    return result
