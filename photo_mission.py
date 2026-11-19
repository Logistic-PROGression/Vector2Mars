#!/usr/bin/env python3
"""Автоматическая фото-миссия Пионер Мини 2.

Порядок действий:
  1. камера переводится вниз (по умолчанию -90°, максимально вниз);
  2. дрон взлетает;
  3. набор высоты над точкой взлёта (по умолчанию 2.7 м);
  4. перелёт в точку (по умолчанию 1.3, 1.3 м от точки взлёта);
  5. пауза на стабилизацию;
  6. фото в --photo-dir;
  7. снижение и полная посадка в этой точке.

Наклон сервопривода работает только на борту (Unix-сокет /tmp/servo.sock),
поэтому для наклона камеры запускайте с --local. С ноутбука миссия всё равно
выполнится, но скрипт честно предупредит, что камера осталась в текущем
положении.

Все шаги, прогресс и ошибки пишутся в JSONL-журнал (--log-dir), фото — в
--photo-dir. По умолчанию журнал: logs/flight_ГГГГММДД_ЧЧММСС.jsonl.

Примеры:
  python3 photo_mission.py --dry-run                  # проверка без железа
  python3 photo_mission.py --local                    # на борту, камера вниз
  python3 photo_mission.py --local --altitude 3.0     # другая высота
  python3 photo_mission.py --local --target-x 1.0 --target-y 1.0
  python3 photo_mission.py --local --tilt -60         # снимать не строго вниз

Коды возврата: 0 — миссия и фото успешны; 1 — ошибка миссии или нет фото;
2 — неверные параметры или нет связи; 130 — прервано Ctrl+C (посадка выполнена).
"""

from __future__ import annotations

import argparse
import sys
import time

from mars.drone import (
    DEFAULT_CAMERA,
    DEFAULT_TCP,
    DroneLink,
    DroneUnavailable,
    SimulatedLink,
)
from mars.flightlog import FlightLogger
from mars.mission import MissionConfig, run_photo_mission

# ---------------------------------------------------------------------------
# Все флаги скрипта (значение по умолчанию в скобках):
#
#   --altitude 2.7          высота над точкой взлёта, м — главный параметр
#   --max-altitude 3.0      защита: запрещает задать высоту выше этого предела
#   --target-x 1.3          куда лететь от точки взлёта по x, м
#   --target-y 1.3          куда лететь от точки взлёта по y, м
#   --tilt -90              физический угол камеры: -90 вниз, 0 горизонт
#   --servo-zero 30         угол SDK, при котором камера смотрит в горизонт
#   --servo-down -80        угол SDK, при котором камера смотрит вниз
#   --servo-priority high   приоритет команд сервопривода: low|medium|high
#   --climb-speed 0.35      максимальная скорость набора высоты, м/с
#   --descend-speed 0.4     максимальная скорость снижения, м/с
#   --move-speed 0.4        максимальная скорость горизонтального перелёта, м/с
#   --move-tolerance 0.08   допуск по горизонтали, м
#   --move-timeout 60       предел времени перелёта в точку, с
#   --tolerance 0.05        допуск по высоте, м (когда считаем цель достигнутой)
#   --settle 1.5            пауза на стабилизацию перед фото, с
#   --tilt-settle 1.0       пауза после команды наклона камеры, с
#   --landing-height 0.25   высота над землёй, с которой начинается посадка, м
#   --photo-retries 3       сколько раз пытаться получить кадр
#   --photo-delay 0.7       пауза между попытками фото, с
#   --climb-timeout 60      предел времени набора высоты, с
#   --descend-timeout 60    предел времени снижения, с
#   --tcp 10.42.0.1:20556   адрес Pioneer SDK 2
#   --camera 10.42.0.1:8554 RTSP-поток камеры
#   --local                 запуск на борту дрона (нужен для наклона камеры)
#   --no-camera             не подключать камеру (фото тогда не будет)
#   --dry-run               симулятор без дрона и SDK
#   --photo-dir photos      куда сохранять фото
#   --log-dir logs          куда писать JSONL-журнал
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    defaults = MissionConfig()
    parser = argparse.ArgumentParser(
        description="Фото-миссия: камера вниз, взлёт, высота, перелёт в точку, фото, посадка.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--altitude",
        type=float,
        default=defaults.altitude_m,
        metavar="М",
        help="высота над точкой взлёта",
    )
    parser.add_argument(
        "--max-altitude",
        type=float,
        default=defaults.max_altitude_m,
        metavar="М",
        help="защита: максимально допустимая высота миссии",
    )
    parser.add_argument(
        "--target-x",
        type=float,
        default=defaults.target_x_m,
        metavar="М",
        help="куда лететь от точки взлёта по x",
    )
    parser.add_argument(
        "--target-y",
        type=float,
        default=defaults.target_y_m,
        metavar="М",
        help="куда лететь от точки взлёта по y",
    )
    parser.add_argument(
        "--tilt",
        type=float,
        default=defaults.tilt_deg,
        metavar="ГРАД",
        help="физический угол камеры: -90 вниз, 0 горизонт",
    )
    parser.add_argument(
        "--servo-zero",
        type=float,
        default=defaults.servo_zero_deg,
        metavar="ГРАД",
        help="угол SDK для горизонта камеры",
    )
    parser.add_argument(
        "--servo-down",
        type=float,
        default=defaults.servo_down_deg,
        metavar="ГРАД",
        help="угол SDK для направления камеры вниз",
    )
    parser.add_argument(
        "--servo-priority",
        choices=("low", "medium", "high"),
        default=defaults.servo_priority,
        help="приоритет команд сервопривода",
    )
    parser.add_argument(
        "--climb-speed",
        type=float,
        default=defaults.climb_speed_ms,
        metavar="М/С",
        help="максимальная скорость набора высоты",
    )
    parser.add_argument(
        "--descend-speed",
        type=float,
        default=defaults.descend_speed_ms,
        metavar="М/С",
        help="максимальная скорость снижения",
    )
    parser.add_argument(
        "--move-speed",
        type=float,
        default=defaults.move_speed_ms,
        metavar="М/С",
        help="максимальная скорость горизонтального перелёта",
    )
    parser.add_argument(
        "--move-tolerance",
        type=float,
        default=defaults.move_tolerance_m,
        metavar="М",
        help="допуск по горизонтали",
    )
    parser.add_argument(
        "--move-timeout",
        type=float,
        default=defaults.move_timeout_s,
        metavar="С",
        help="предел времени перелёта в точку",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=defaults.tolerance_m,
        metavar="М",
        help="допуск по высоте",
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=defaults.settle_s,
        metavar="С",
        help="пауза на стабилизацию перед фото",
    )
    parser.add_argument(
        "--tilt-settle",
        type=float,
        default=defaults.tilt_settle_s,
        metavar="С",
        help="пауза после команды наклона камеры",
    )
    parser.add_argument(
        "--landing-height",
        type=float,
        default=defaults.landing_height_m,
        metavar="М",
        help="высота над землёй, с которой начинается посадка",
    )
    parser.add_argument(
        "--photo-retries",
        type=int,
        default=defaults.photo_retries,
        metavar="N",
        help="число попыток получить кадр",
    )
    parser.add_argument(
        "--photo-delay",
        type=float,
        default=defaults.photo_delay_s,
        metavar="С",
        help="пауза между попытками фото",
    )
    parser.add_argument(
        "--climb-timeout",
        type=float,
        default=defaults.climb_timeout_s,
        metavar="С",
        help="предел времени набора высоты",
    )
    parser.add_argument(
        "--descend-timeout",
        type=float,
        default=defaults.descend_timeout_s,
        metavar="С",
        help="предел времени снижения",
    )
    parser.add_argument(
        "--tcp",
        default=DEFAULT_TCP,
        help="адрес Pioneer SDK 2",
    )
    parser.add_argument(
        "--camera",
        default=DEFAULT_CAMERA,
        help="RTSP-поток камеры",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="запуск на борту дрона (нужен для наклона камеры)",
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="не подключать камеру (фото тогда не будет)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="симуляция без дрона и SDK",
    )
    parser.add_argument(
        "--photo-dir",
        default="photos",
        help="куда сохранять фото",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="куда писать журнал полёта",
    )
    return parser.parse_args(argv)


def _notice(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config = MissionConfig(
            altitude_m=args.altitude,
            max_altitude_m=args.max_altitude,
            target_x_m=args.target_x,
            target_y_m=args.target_y,
            tilt_deg=args.tilt,
            servo_zero_deg=args.servo_zero,
            servo_down_deg=args.servo_down,
            servo_priority=args.servo_priority,
            climb_speed_ms=args.climb_speed,
            descend_speed_ms=args.descend_speed,
            move_speed_ms=args.move_speed,
            move_tolerance_m=args.move_tolerance,
            tolerance_m=args.tolerance,
            settle_s=args.settle,
            tilt_settle_s=args.tilt_settle,
            landing_height_m=args.landing_height,
            photo_retries=args.photo_retries,
            photo_delay_s=args.photo_delay,
            climb_timeout_s=args.climb_timeout,
            descend_timeout_s=args.descend_timeout,
            move_timeout_s=args.move_timeout,
        )
    except ValueError as exc:
        print(f"Ошибка параметров: {exc}", file=sys.stderr)
        return 2

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
        try:
            link.connect()
        except DroneUnavailable as exc:
            logger.log("connect_failed", error=str(exc))
            print(f"Не удалось подключиться: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:  # pragma: no cover - defensive
            logger.log("connect_failed", error=str(exc))
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
            _notice("Камера недоступна: " + link.camera_error)

        result = run_photo_mission(
            link, config, logger, args.photo_dir, notice=_notice
        )
    except KeyboardInterrupt:
        logger.log("interrupted")
        print("Прервано оператором", file=sys.stderr)
        return 130
    finally:
        try:
            link.close()
        except Exception:
            pass
        logger.close()

    if result.interrupted:
        return 130
    if result.ok:
        _notice(f"Готово. Фото: {result.photo}  Журнал: {logger.path}")
        return 0
    _notice(f"Миссия не выполнена: {result.reason or 'фото не получено'}")
    _notice(f"Журнал: {logger.path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
