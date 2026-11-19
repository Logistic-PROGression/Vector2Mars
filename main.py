#!/usr/bin/env python3
"""Analyze a saved camera frame and report the shortest safe route.

Usage:
    python3 main.py --frame PATH [--config config/routes.json] [--json]
                    [--debug-out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from mars.config import load_config
from mars.vision import RouteMeasurement, SceneResult, analyze_frame, annotate

DEFAULT_CONFIG = Path("config/routes.json")

# Russian names for the colours used in the task output. Unknown names pass
# through unchanged so a custom palette still prints something meaningful.
COLOR_RU: dict[str, str] = {
    "red": "красный",
    "green": "зелёный",
    "blue": "синий",
    "yellow": "жёлтый",
    "orange": "оранжевый",
    "purple": "фиолетовый",
    "white": "белый",
    "black": "чёрный",
}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Найти кратчайший безопасный маршрут на кадре с камеры БПЛА."
    )
    parser.add_argument("--frame", required=True, help="путь к кадру (JPEG/PNG)")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"путь к JSON-конфигурации (по умолчанию {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="вывести результат в машиночитаемом JSON",
    )
    parser.add_argument(
        "--debug-out",
        default=None,
        help="сохранить изображение с разметкой по указанному пути",
    )
    return parser.parse_args(argv)


def _measurement_to_dict(route: RouteMeasurement) -> dict:
    return {
        "name": route.name,
        "number": route.number,
        "length_px": round(float(route.length_px), 2),
        "length_m": None if route.length_m is None else round(float(route.length_m), 4),
        "area_px": int(route.area_px),
        "dangerous": bool(route.dangerous),
    }


def _result_to_dict(result: SceneResult) -> dict:
    return {
        "routes": [_measurement_to_dict(route) for route in result.routes],
        "shortest": (
            None if result.shortest is None else _measurement_to_dict(result.shortest)
        ),
        "excluded": [_measurement_to_dict(route) for route in result.excluded],
    }


def _color_and_number(route: RouteMeasurement) -> str:
    color = COLOR_RU.get(route.name.lower(), route.name)
    if route.number is None:
        return color
    return f"{color}, №{route.number}"


def _describe(route: RouteMeasurement) -> str:
    if route.length_m is not None:
        length = f"{route.length_m:.2f} м"
    else:
        length = f"{route.length_px:.0f} px (масштаб не задан)"
    return f"{_color_and_number(route)}, длина {length}"


def _write_debug_image(frame, result: SceneResult, destination: str) -> None:
    path = Path(destination).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), annotate(frame, result)):
            print(f"Не удалось записать отладочное изображение: {path}", file=sys.stderr)
    except OSError as exc:
        print(f"Не удалось записать отладочное изображение: {path}: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        config = load_config(args.config)
    except (OSError, ValueError) as exc:
        print(f"Не удалось загрузить конфигурацию: {exc}", file=sys.stderr)
        return 2

    try:
        frame = cv2.imread(str(args.frame))
    except cv2.error:
        frame = None
    if frame is None:
        print(f"Не удалось прочитать кадр: {args.frame}", file=sys.stderr)
        return 2

    result = analyze_frame(frame, config)

    if args.debug_out:
        _write_debug_image(frame, result, args.debug_out)

    if args.json:
        payload = _result_to_dict(result)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        if result.shortest is None:
            print("Маршруты не найдены.")
        else:
            print(f"Кратчайший маршрут: {_describe(result.shortest)}")
        for route in result.excluded:
            print(f"Исключён опасный маршрут: {_color_and_number(route)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
