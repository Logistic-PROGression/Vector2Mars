#!/usr/bin/env python3
"""Автоматическая фото-миссия Пионер Мини 2.

Порядок действий:
  1. камера переводится вниз (по умолчанию -90°, максимально вниз);
  2. дрон взлетает;
  3. набор высоты над точкой взлёта (по умолчанию 2.7 м);
  4. перелёт к точке (по умолчанию 0.5, 0.5 м от взлёта) с 5 остановками
     (0, 25, 50, 75 и 100% пути); на каждой — пауза на стабилизацию и фото;
  5. снижение и полная посадка в конечной точке;
  6. после посадки кадры анализируются: кадры без меток отбрасываются,
     остальные совмещаются по центрам ArUco-меток, стрелки голосуются по
     каждой позиции (2 из 3, 3 из 5), поэтому потеря одной стрелки на кадре
     не портит счёт; маршруты выводятся по возрастанию длины.

Флаг --expect-arrows тихо оставляет для сшивки только кадры, где набор чисел
стрелок на маршрутах совпал с ожидаемым (без чисел — 13 8 3). Если совпадений
нет, анализ идёт как без флага; в консоль ничего дополнительно не печатается.

Флаги можно включить и в самом коде: отредактируйте CODE_DEFAULTS в начале
файла, например CodeDefaults(annotate_all=True, expect_arrows=(13, 8, 3)).
Числовые параметры миссии при этом задаются в MissionConfig.

Метки должны быть видны целиком хотя бы на двух кадрах: по четырём центрам
строится гомография, по углам меток она проверяется. Если сшить кадры нельзя,
скрипт откатывается на лучший одиночный кадр.

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
  python3 photo_mission.py --local --quadrant sw      # лететь в −x −y
  python3 photo_mission.py --local --quadrant se --target-x 1.0 --target-y 1.0
  python3 photo_mission.py --local --tilt -60         # снимать не строго вниз

Коды возврата: 0 — миссия и фото успешны; 1 — ошибка миссии или нет фото;
2 — неверные параметры или нет связи; 130 — прервано Ctrl+C (посадка выполнена).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from mars.drone import (
    DEFAULT_CAMERA,
    DEFAULT_TCP,
    DroneLink,
    DroneUnavailable,
    SimulatedLink,
)
from mars.flightlog import FlightLogger
from mars.mission import MissionConfig, run_photo_mission

# Arrow counts of the three routes on the competition field. Used only when
# --expect-arrows is given without numbers.
DEFAULT_EXPECTED_ARROWS = (13, 8, 3)


@dataclass(frozen=True)
class CodeDefaults:
    """Flag defaults that can be set in code instead of the command line.

    Change ``CODE_DEFAULTS`` below to enable a mode without arguments, for
    example ``CodeDefaults(annotate_all=True, expect_arrows=(13, 8, 3))``.
    Numeric mission parameters stay in :class:`mars.mission.MissionConfig`.
    """

    local: bool = False
    no_camera: bool = False
    dry_run: bool = False
    annotate_all: bool = False
    expect_arrows: tuple[int, ...] | None = None


# These values are used when the matching flag is absent from the command
# line. Edit the line below to hardcode an enabled flag.
CODE_DEFAULTS = CodeDefaults()

# ---------------------------------------------------------------------------
# Все флаги скрипта (значение по умолчанию в скобках).
# Переключатели можно включить заранее в CODE_DEFAULTS выше:
#   CODE_DEFAULTS = CodeDefaults(annotate_all=True, expect_arrows=(13, 8, 3))
#
#   --altitude 2.7          высота над точкой взлёта, м — главный параметр
#   --max-altitude 3.0      защита: запрещает задать высоту выше этого предела
#   --target-x 0.5          куда лететь от точки взлёта по x, м
#   --target-y 0.5          куда лететь от точки взлёта по y, м
#   --quadrant auto         направление полёта: auto|ne|nw|se|sw
#                           (ne=+x+y, nw=-x+y, se=+x-y, sw=-x-y;
#                            auto — знаки берутся из --target-x/--target-y)
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
#   --stops 5               сколько остановок с фото на пути (5 → 0, 25, 50, 75, 100%)
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
#   --annotate-all          разметить не только итоговый, но и все кадры
#   --expect-arrows         проверить число стрелок на маршрутах
#                           (без чисел — 13 8 3; без флага проверка выключена)
# ---------------------------------------------------------------------------


def _expected_arrows(values: list[int] | None) -> tuple[int, ...] | None:
    """Turn the optional flag into a hint; None keeps the check disabled."""
    if values is None:
        return None
    return tuple(values) if values else DEFAULT_EXPECTED_ARROWS


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    defaults = MissionConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Фото-миссия: камера вниз, взлёт, высота, перелёт с остановками, "
            "фото, посадка."
        ),
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
        "--quadrant",
        choices=("auto", "ne", "nw", "se", "sw"),
        default="auto",
        help=(
            "направление полёта: ne=+x+y, nw=-x+y, se=+x-y, sw=-x-y; "
            "auto — знаки берутся из --target-x/--target-y"
        ),
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
        "--stops",
        type=int,
        default=defaults.stop_count,
        metavar="N",
        help="число остановок с фото на пути: 5 — равные шаги от взлёта до цели",
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
        default=CODE_DEFAULTS.local,
        help="запуск на борту дрона (нужен для наклона камеры)",
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        default=CODE_DEFAULTS.no_camera,
        help="не подключать камеру (фото тогда не будет)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=CODE_DEFAULTS.dry_run,
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
    parser.add_argument(
        "--annotate-all",
        action="store_true",
        default=CODE_DEFAULTS.annotate_all,
        help="разметить все кадры, а не только итоговый",
    )
    parser.add_argument(
        "--expect-arrows",
        nargs="*",
        type=int,
        default=(
            None
            if CODE_DEFAULTS.expect_arrows is None
            else list(CODE_DEFAULTS.expect_arrows)
        ),
        metavar="N",
        help=(
            "проверить число стрелок на маршрутах; без чисел — 13 8 3, "
            "без флага проверка выключена"
        ),
    )
    return parser.parse_args(argv)


def _notice(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


def _analyse_photos(
    photos: list[Path],
    logger: FlightLogger,
    *,
    annotate_all: bool = False,
    expect_arrows: tuple[int, ...] | None = None,
) -> None:
    """Mark the stop photos, fuse them and report the routes.

    Every readable frame is analysed. Frames without markers are ignored: the
    field is already out of the picture there. The remaining frames are
    aligned by their ArUco centres and vote per arrow, so one frame that lost
    an arrow or invented a bright patch cannot change the result. If the
    frames cannot be aligned, the best single frame is used. Routes are
    printed shortest first.

    With ``expect_arrows`` only full-marker frames whose route arrow counts
    match the expected set are fused. No match keeps every frame, so the
    report is exactly the same as without the hint.
    """
    try:
        import cv2  # noqa: PLC0415 - optional heavy dependency

        from mars.aruco import detect_markers  # noqa: PLC0415
        from mars.arrow_routes import (  # noqa: PLC0415
            COLOUR_BLUE,
            COLOUR_LABELS_RU,
            COLOUR_RED,
            COLOUR_WHITE,
            annotate_arrow_routes,
            detect_arrows,
            routes_from_arrows,
            second_longest_route,
        )
        from mars.fusion import (  # noqa: PLC0415
            FrameObservation,
            fuse_routes,
            select_by_arrow_hint,
        )
    except Exception as exc:  # noqa: BLE001 - never fail the mission on vision
        logger.log("route_error", error=str(exc))
        _notice(f"Маршруты не посчитаны: {exc}")
        return

    observations: list[FrameObservation] = []
    for photo in photos:
        try:
            frame = cv2.imread(str(photo))
        except cv2.error:
            frame = None
        if frame is None:
            logger.log("frame_unreadable", photo=str(photo))
            continue
        try:
            markers = detect_markers(frame)
            arrows = detect_arrows(frame, markers)
            routes = routes_from_arrows(
                arrows,
                markers,
                frame_shape=(frame.shape[0], frame.shape[1]),
            )
        except Exception as exc:  # noqa: BLE001 - one bad frame is not fatal
            logger.log("frame_error", photo=str(photo), error=str(exc))
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        observations.append(
            FrameObservation(
                path=Path(photo),
                image=frame,
                markers=tuple(markers),
                arrows=tuple(arrows),
                routes=tuple(routes),
                sharpness=sharpness,
            )
        )
        logger.log(
            "frame_analysed",
            photo=str(photo),
            markers=len(markers),
            routes=len(routes),
            arrows=sum(len(route.arrows) for route in routes),
            candidates=len(arrows),
            sharpness=round(sharpness, 1),
        )

    if not observations:
        _notice("Кадры не прочитаны — маршруты не посчитаны")
        return

    if expect_arrows is not None:
        selected = select_by_arrow_hint(observations, expect_arrows)
        # The hint is intentionally silent in the console; the JSONL log keeps
        # enough detail to see whether the filter changed anything.
        logger.log(
            "arrow_hint",
            expected=list(expect_arrows),
            matched=selected is not observations,
            frames=[str(observation.path) for observation in selected],
        )
        observations = selected

    fused = fuse_routes(observations)
    reference = fused.reference
    if reference is None:
        _notice("Метки не найдены — маршруты не посчитаны")
        return

    if fused.mode == "fusion":
        counts = ", ".join(str(count) for count in fused.frame_counts)
        line = (
            f"Сшито кадров {len(fused.used)} ({counts}) → "
            f"подтверждено {fused.confirmed}"
        )
        if fused.dropped:
            line += f", отброшено {fused.dropped}"
        _notice(line)
    else:
        _notice(f"Сшивка не удалась ({fused.note}); беру лучший кадр")

    ranked = sorted(fused.routes, key=lambda route: route.length_px)
    second = second_longest_route(ranked)

    frame_name = reference.path.name if reference.path is not None else "?"
    _notice(
        f"Размечен кадр {frame_name}: меток {len(reference.markers)}, "
        f"маршрутов {len(ranked)}"
    )
    for rank, route in enumerate(ranked, 1):
        mixed = Counter(route.colors)
        breakdown = ", ".join(
            f"{COLOUR_LABELS_RU[color]} {mixed[color]}"
            for color in (COLOUR_RED, COLOUR_BLUE, COLOUR_WHITE)
            if mixed[color]
        )
        _notice(
            f"{rank}. {route.start_marker_id} -> {route.end_marker_id}: "
            f"стрелок {len(route.arrows)} ({breakdown}), "
            f"{route.length_px:.0f} px"
        )

    summary = [
        {
            "start": route.start_marker_id,
            "end": route.end_marker_id,
            "arrows": len(route.arrows),
            "colors": dict(Counter(route.colors)),
            "length_px": round(route.length_px, 1),
        }
        for route in ranked
    ]
    logger.log(
        "routes",
        frame=str(reference.path) if reference.path is not None else None,
        mode=fused.mode,
        note=fused.note,
        used_frames=[
            str(observation.path) for observation in fused.used
        ],
        frame_counts=list(fused.frame_counts),
        confirmed=fused.confirmed,
        dropped=fused.dropped,
        markers=[marker.id for marker in reference.markers],
        count=len(ranked),
        second=None if second is None else summary[ranked.index(second)],
        routes=summary,
    )

    try:
        overlay = annotate_arrow_routes(
            reference.image, list(fused.routes), list(reference.markers)
        )
        if reference.path is None:
            raise ValueError("у опорного кадра нет пути")
        annotated = reference.path.with_name(
            reference.path.stem + "_annotated.png"
        )
        if cv2.imwrite(str(annotated), overlay):
            _notice(f"Разметка сохранена: {annotated}")
        else:
            logger.log("annotate_failed", photo=str(annotated))
    except Exception as exc:  # noqa: BLE001 - annotation is a bonus
        logger.log("annotate_error", error=str(exc))

    if annotate_all:
        marked = 0
        for observation in observations:
            try:
                if observation.path is None:
                    continue
                overlay = annotate_arrow_routes(
                    observation.image,
                    list(observation.routes),
                    list(observation.markers),
                )
                annotated = observation.path.with_name(
                    observation.path.stem + "_annotated_frame.png"
                )
                if cv2.imwrite(str(annotated), overlay):
                    marked += 1
                else:
                    logger.log("annotate_failed", photo=str(annotated))
            except Exception as exc:  # noqa: BLE001 - annotation is a bonus
                logger.log(
                    "annotate_error",
                    photo=str(observation.path),
                    error=str(exc),
                )
        _notice(f"Размечено исходных кадров: {marked}")

    if second is None:
        _notice("Маршрутов меньше двух — второй по длине не определён")
    else:
        _notice(
            f"2-й по длине: {second.start_marker_id}->{second.end_marker_id}, "
            f"стрелок {len(second.arrows)}, {second.length_px:.0f} px"
        )


_QUADRANT_SIGNS = {
    "ne": (1, 1),
    "nw": (-1, 1),
    "se": (1, -1),
    "sw": (-1, -1),
}


def _target_with_quadrant(args: argparse.Namespace) -> tuple[float, float]:
    """Return target x/y with the sign selected by ``--quadrant``.

    ``auto`` keeps the signs of ``--target-x``/``--target-y`` untouched.
    """
    if args.quadrant == "auto":
        return args.target_x, args.target_y
    sign_x, sign_y = _QUADRANT_SIGNS[args.quadrant]
    return abs(args.target_x) * sign_x, abs(args.target_y) * sign_y


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    target_x, target_y = _target_with_quadrant(args)
    try:
        config = MissionConfig(
            altitude_m=args.altitude,
            max_altitude_m=args.max_altitude,
            target_x_m=target_x,
            target_y_m=target_y,
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
            stop_count=args.stops,
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
        if result.photos:
            _analyse_photos(
                result.photos,
                logger,
                annotate_all=args.annotate_all,
                expect_arrows=_expected_arrows(args.expect_arrows),
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
    _notice(f"Журнал: {logger.path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
