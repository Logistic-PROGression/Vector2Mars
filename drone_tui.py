#!/usr/bin/env python3
"""Fly a Pioneer Mini 2 from a laptop over Wi-Fi.

The TUI sends body-fixed velocity commands, shows live telemetry, tilts the
camera servo (on board only), saves photos and writes a JSONL flight log.

Usage:
    python3 drone_tui.py --dry-run          # no hardware, demo the interface
    python3 drone_tui.py                    # laptop on the drone Wi-Fi
    python3 drone_tui.py --local            # run on the drone itself

Safety: releasing a key stops that axis after 0.35 s, ``X`` lands before exit,
and the altitude guard refuses to climb above ``--max-altitude``.
"""

from __future__ import annotations

import argparse
import curses
import locale
import sys
import time
from collections import deque

from mars.drone import (
    DEFAULT_CAMERA,
    DEFAULT_TCP,
    DroneLink,
    DroneUnavailable,
    SimulatedLink,
    Telemetry,
)
from mars.flightlog import FlightLogger, FlightTrack
from mars.teleop import KEY_COMMANDS, KEY_HELP, ControlState, Velocity

REFRESH_S = 0.1
TELEMETRY_S = 0.5
TELEMETRY_LOG_S = 1.0
SERVO_STEP = 10.0
EVENT_LINES = 6


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Пульт управления Пионер Мини 2 в терминале."
    )
    parser.add_argument("--tcp", default=DEFAULT_TCP, help=f"адрес SDK (по умолчанию {DEFAULT_TCP})")
    parser.add_argument("--camera", default=DEFAULT_CAMERA, help=f"RTSP-поток камеры (по умолчанию {DEFAULT_CAMERA})")
    parser.add_argument("--local", action="store_true", help="запуск на борту дрона, без сети")
    parser.add_argument("--no-camera", action="store_true", help="не подключать камеру")
    parser.add_argument("--dry-run", action="store_true", help="симуляция без дрона и SDK")
    parser.add_argument("--speed", type=float, default=0.5, help="скорость, м/с (по умолчанию 0.5)")
    parser.add_argument("--max-speed", type=float, default=1.5, help="предел скорости, м/с")
    parser.add_argument("--max-altitude", type=float, default=3.0, help="защита по высоте, м")
    parser.add_argument(
        "--servo-priority",
        choices=("high", "medium", "low"),
        default="high",
        help="приоритет команд сервопривода камеры (по умолчанию high)",
    )
    parser.add_argument(
        "--servo-test",
        action="store_true",
        help="прогнать сервопривод -80..30 и выйти (диагностика)",
    )
    parser.add_argument("--photo-dir", default="photos", help="куда сохранять фото")
    parser.add_argument("--log-dir", default="logs", help="куда писать журнал полёта")
    return parser.parse_args(argv)


def _fmt(value: float | None, suffix: str = "", digits: int = 1) -> str:
    return "—" if value is None else f"{value:.{digits}f}{suffix}"


class Tui:
    def __init__(
        self,
        screen,
        link: DroneLink,
        logger: FlightLogger,
        args: argparse.Namespace,
    ) -> None:
        self.screen = screen
        self.link = link
        self.logger = logger
        self.args = args
        self.state = ControlState(
            speed=args.speed,
            max_speed=args.max_speed,
        )
        self.telemetry = Telemetry()
        self.track = FlightTrack()
        self.running = True
        self.help_visible = False
        self.preview = False
        self.camera_angle = 0.0
        self.servo_target: float | None = None
        self._last_servo_send = 0.0
        self.events: deque[str] = deque(maxlen=EVENT_LINES)
        self.message = "Готов. T — взлёт, H — справка."
        self._last_telemetry = 0.0
        self._last_telemetry_log = 0.0
        self._last_send = 0.0
        self._last_velocity = None
        self._last_command: str | None = None
        self.velocity = Velocity()
        self._altitude_blocked = False

    # -- lifecycle ---------------------------------------------------------

    def run(self) -> None:
        self.screen.keypad(True)
        self.screen.timeout(int(REFRESH_S * 1000))
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        self._init_colors()
        if self.link.camera_error:
            self._note("Камера недоступна: " + self.link.camera_error)
        else:
            self._note("TUI запущен")
        self.logger.log("tui_started", dry_run=self.args.dry_run, local=self.args.local)
        try:
            while self.running:
                self._handle_key()
                now = time.monotonic()
                self._update(now)
                self._draw()
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        self.state.stop()
        try:
            self.link.stop()
        except Exception:
            pass
        try:
            state = self.link.telemetry().state
        except Exception:
            state = "UNKNOWN"
        if state == "IN_SKY":
            self._note("Автоматическая посадка")
            self.logger.log("auto_land")
            try:
                self.link.land()
            except Exception as exc:
                self.logger.log("auto_land_failed", error=str(exc))
        if self.track.samples:
            self.logger.log("flight_summary", **self.track.summary())
        self.logger.log("tui_stopped")

    # -- input -------------------------------------------------------------

    def _handle_key(self) -> None:
        key = self.screen.getch()
        if key == -1:
            return
        if key == 27:  # Esc
            self.running = False
            return
        try:
            name = curses.keyname(key).decode("utf-8", "replace")
        except Exception:
            return
        if not name.startswith("KEY_"):
            name = name.lower()

        command = KEY_COMMANDS.get(name)
        if command is not None:
            self.state.press(command, time.monotonic())
            if command != self._last_command:
                self._last_command = command
                self.logger.log("command", command=command)
            return

        if name in ("t",):
            self._takeoff()
        elif name in ("l",):
            self._land()
        elif name in ("p",):
            self._photo()
        elif name == "[":
            self._tilt(-SERVO_STEP)
        elif name == "]":
            self._tilt(+SERVO_STEP)
        elif name in ("1", "2", "3", "4"):
            self._set_tilt({"1": -80.0, "2": -40.0, "3": 0.0, "4": 30.0}[name])
        elif name in ("+", "="):
            self._change_speed(+0.1)
        elif name in ("-", "_"):
            self._change_speed(-0.1)
        elif name in ("v",):
            self.preview = not self.preview
            self._note("Окно камеры включено" if self.preview else "Окно камеры выключено")
        elif name in ("h",):
            self.help_visible = not self.help_visible
        elif name == "x":
            self.running = False

    def _takeoff(self) -> None:
        self._note("Взлёт…")
        self.logger.log("takeoff_requested")
        try:
            ground_altitude = self.link.telemetry().altitude_m
        except Exception:
            ground_altitude = None
        try:
            ok = self.link.takeoff()
        except Exception as exc:
            ok = False
            self.logger.log("takeoff_error", error=str(exc))
        self.logger.log("takeoff", ok=bool(ok))
        if ok:
            self.track = FlightTrack()
            self.track.mark_takeoff(ground_altitude)
            try:
                altitude = self.link.telemetry().altitude_m
            except Exception:
                altitude = None
            self.logger.log(
                "takeoff_complete",
                altitude_m=altitude,
                ground_altitude_m=ground_altitude,
            )
        self._note("Взлёт выполнен" if ok else "Взлёт не подтверждён")

    def _land(self) -> None:
        self.state.stop()
        self._note("Посадка…")
        self.logger.log("land_requested")
        try:
            ok = self.link.land()
        except Exception as exc:
            ok = False
            self.logger.log("land_error", error=str(exc))
        self.logger.log("land", ok=bool(ok))
        if ok:
            try:
                altitude = self.link.telemetry().altitude_m
            except Exception:
                altitude = None
            self.logger.log("land_complete", altitude_m=altitude)
        self._note("Посадка выполнена" if ok else "Посадка не подтверждена")

    def _photo(self) -> None:
        try:
            path = self.link.take_photo(self.args.photo_dir)
        except Exception as exc:
            self.logger.log("photo_error", error=str(exc))
            self._note(f"Фото не сохранено: {exc}")
            return
        if path is None:
            self.logger.log("photo_failed", camera_error=self.link.camera_error)
            if self.link.camera_error:
                self._note("Камера недоступна: " + self.link.camera_error)
            else:
                self._note("Кадр не получен: проверьте камеру и поток")
            return
        self.logger.log("photo", path=str(path))
        self._note(f"Фото: {path}")

    def _tilt(self, delta: float) -> None:
        self._set_tilt(self.camera_angle + delta)

    def _set_tilt(self, target: float) -> None:
        target = max(-80.0, min(30.0, target))
        try:
            ok = self.link.set_camera_angle(target, priority=self.args.servo_priority)
        except DroneUnavailable as exc:
            self._note(str(exc))
            self.logger.log("tilt_unavailable", error=str(exc))
            return
        except Exception as exc:
            self._note(f"Камера: {exc}")
            return
        if ok:
            self.camera_angle = target
            self.servo_target = target
            self._last_servo_send = time.monotonic()
            self.logger.log(
                "camera_angle", angle=target, priority=self.args.servo_priority
            )
            self._note(f"Камера: {target:.0f}°")
        else:
            self.logger.log(
                "camera_angle_failed", angle=target, priority=self.args.servo_priority
            )
            self._note("Сервопривод не подтвердил команду")

    def _change_speed(self, delta: float) -> None:
        speed = self.state.change_speed(delta)
        self.logger.log("speed", speed=round(speed, 2))
        self._note(f"Скорость: {speed:.1f} м/с")

    # -- loop --------------------------------------------------------------

    def _update(self, now: float) -> None:
        if now - self._last_telemetry >= TELEMETRY_S:
            self._last_telemetry = now
            try:
                self.telemetry = self.link.telemetry()
            except Exception as exc:
                self.logger.log("telemetry_error", error=str(exc))

        if now - self._last_telemetry_log >= TELEMETRY_LOG_S:
            self._last_telemetry_log = now
            self.track.add(self.telemetry)
            self.logger.log(
                "telemetry",
                **self.telemetry.as_dict(),
                altitude_gain_m=self.track.altitude_gain(),
                path_length_m=round(self.track.path_length_m, 2),
            )

        velocity = self.state.tick(now)
        self.velocity = velocity
        if self.telemetry.state == "IN_SKY" and now - self._last_send >= REFRESH_S:
            self._last_send = now
            vz = velocity.vz
            position = self.telemetry.position
            if position is not None and position[2] >= self.args.max_altitude and vz > 0:
                vz = 0.0
                if not self._altitude_blocked:
                    self._altitude_blocked = True
                    self._note("Предел высоты: набор высоты заблокирован")
            elif vz <= 0:
                self._altitude_blocked = False
            try:
                self.link.set_velocity(
                    velocity.vx, velocity.vy, vz, velocity.yaw_rate, REFRESH_S * 2
                )
            except Exception as exc:
                self.logger.log("velocity_error", error=str(exc))
            if velocity != self._last_velocity:
                self._last_velocity = velocity
                self.logger.log(
                    "velocity",
                    vx=round(velocity.vx, 2),
                    vy=round(velocity.vy, 2),
                    vz=round(vz, 2),
                    yaw_rate=round(velocity.yaw_rate, 1),
                )

        # Hold the commanded tilt: the board may otherwise re-stabilise the
        # gimbal and silently undo a single manual command.
        if self.servo_target is not None and now - self._last_servo_send >= 1.0:
            self._last_servo_send = now
            try:
                self.link.set_camera_angle(
                    self.servo_target, priority=self.args.servo_priority, attempts=1
                )
            except Exception:
                pass

        if self.preview:
            frame = self.link.grab_frame(timeout=0.2)
            if frame is not None:
                try:
                    import cv2  # noqa: PLC0415 - optional GUI preview

                    cv2.imshow("Pioneer Mini 2", frame)
                    cv2.waitKey(1)
                except Exception:
                    self.preview = False
                    self._note("Окно камеры недоступно (нет GUI)")

    # -- rendering ---------------------------------------------------------

    def _init_colors(self) -> None:
        if not curses.has_colors():
            return
        curses.start_color()
        try:
            curses.use_default_colors()
            background = -1
        except curses.error:
            background = curses.COLOR_BLACK
        curses.init_pair(1, curses.COLOR_GREEN, background)
        curses.init_pair(2, curses.COLOR_YELLOW, background)
        curses.init_pair(3, curses.COLOR_RED, background)
        curses.init_pair(4, curses.COLOR_CYAN, background)

    def _color(self, pair: int) -> int:
        return curses.color_pair(pair) if curses.has_colors() else 0

    def _add(self, row: int, text: str, attr: int = 0) -> None:
        height, width = self.screen.getmaxyx()
        if row < 0 or row >= height:
            return
        try:
            self.screen.addstr(row, 0, text[: max(0, width - 1)], attr)
        except curses.error:
            pass

    def _draw(self) -> None:
        self.screen.erase()
        height, width = self.screen.getmaxyx()
        tel = self.telemetry

        title = " Vector2Mars · Pioneer Mini 2 "
        if self.args.dry_run:
            title += "· СИМУЛЯЦИЯ "
        self._add(0, title, curses.A_REVERSE | curses.A_BOLD)

        state_attr = self._color(1) if tel.state == "IN_SKY" else self._color(2)
        self._add(1, f" Состояние: {tel.state}", state_attr | curses.A_BOLD)

        battery = f"Батарея: {_fmt(tel.battery_v, ' В', 2)}"
        if tel.battery_c is not None:
            battery += f"  {tel.battery_c:.0f} °C"
        battery_attr = self._color(3) if (tel.battery_v or 99) < 3.5 else 0
        self._add(2, f" {battery}", battery_attr)

        if tel.position is None:
            position = "Позиция: нет данных LPS"
        else:
            x, y, z = tel.position
            position = f"Позиция: x={x:+.2f}  y={y:+.2f}  z={z:+.2f} м"
        position += f"   Курс: {_fmt(tel.yaw_deg, '°')}"
        self._add(3, f" {position}")

        velocity = self.velocity
        self._add(
            4,
            f" Скорость: vx={velocity.vx:+.2f}  vy={velocity.vy:+.2f}  "
            f"vz={velocity.vz:+.2f}  yaw={velocity.yaw_rate:+.0f}°/с"
            f"   предел {self.state.speed:.1f} м/с",
        )
        self._add(5, " " + "─" * max(0, width - 2), curses.A_DIM)

        row = 6
        if self.help_visible:
            for keys, description in KEY_HELP:
                self._add(row, f"  {keys:<12} {description}")
                row += 1
        else:
            self._add(row, "  W/S A/D — движение   R/F — высота   Q/E — поворот   Пробел — стоп")
            row += 1
            self._add(row, "  T/L — взлёт/посадка   P — фото   [ ] — камера   V — окно камеры   H — справка   X — выход")
            row += 1

        row += 1
        self._add(row, " Журнал:", curses.A_BOLD)
        row += 1
        for line in list(self.events)[-EVENT_LINES:]:
            self._add(row, "  " + line)
            row += 1

        if self.message:
            self._add(max(0, height - 2), " " + self.message, self._color(4))

        if self.logger.error:
            self._add(max(0, height - 1), " Лог: " + self.logger.error, self._color(3))
        else:
            log_name = self.logger.path.name if self.logger.path else "нет"
            self._add(max(0, height - 1), f" Лог: {log_name}", curses.A_DIM)
        self.screen.refresh()

    def _note(self, text: str) -> None:
        self.message = text
        stamp = time.strftime("%H:%M:%S")
        self.events.append(f"{stamp}  {text}")


def _run(screen, link: DroneLink, logger: FlightLogger, args: argparse.Namespace) -> None:
    Tui(screen, link, logger, args).run()


def _servo_test(link: DroneLink, args: argparse.Namespace) -> int:
    """Sweep the servo so a human can see whether it physically moves."""
    print("Тест сервопривода: -80 → +30 → 0, шаг 10°.")
    for angle in [*range(-80, 31, 10), 0]:
        try:
            ok = link.set_camera_angle(angle, priority=args.servo_priority)
        except DroneUnavailable as exc:
            print(f"Сервопривод недоступен: {exc}")
            return 2
        print(f"  {angle:+4d}° — {'команда отправлена' if ok else 'ОШИБКА отправки'}")
        time.sleep(0.8)
    print("Если камера не двигалась, проблема не в TUI: проверьте сервопривод и /tmp/servo.sock.")
    return 0


def main(argv: list[str] | None = None) -> int:
    locale.setlocale(locale.LC_ALL, "")
    args = _parse_args(argv)
    logger = FlightLogger(args.log_dir)

    link: DroneLink
    if args.dry_run:
        link = SimulatedLink()
    else:
        link = DroneLink(
            tcp=args.tcp,
            camera_ip=args.camera,
            local=args.local,
            with_camera=not args.no_camera,
        )

    try:
        link.connect()
    except DroneUnavailable as exc:
        logger.log("connect_failed", error=str(exc))
        logger.close()
        print(f"Не удалось подключиться: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - defensive
        logger.log("connect_failed", error=str(exc))
        logger.close()
        print(f"Не удалось подключиться: {exc}", file=sys.stderr)
        return 2

    logger.log(
        "connected",
        tcp=args.tcp,
        camera=args.camera,
        local=args.local,
        dry_run=args.dry_run,
    )
    if link.camera_error:
        logger.log("camera_unavailable", error=link.camera_error)

    if args.servo_test:
        try:
            return _servo_test(link, args)
        finally:
            link.close()
            logger.close()

    try:
        curses.wrapper(_run, link, logger, args)
    except KeyboardInterrupt:
        pass
    except Exception as exc:  # pragma: no cover - terminal-specific failures
        logger.log("tui_crashed", error=str(exc))
        print(f"Ошибка интерфейса: {exc}", file=sys.stderr)
        return 1
    finally:
        link.close()
        logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
