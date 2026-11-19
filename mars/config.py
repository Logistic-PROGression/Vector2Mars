"""Configuration model for the route-detection pipeline.

All tunable values (colour thresholds, scale, minimum blob size) live in a JSON
file so the palette can be recalibrated on real camera frames without touching
code. ``config/routes.json`` ships with placeholder values.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

HSVTriplet = tuple[int, int, int]
HSVRange = tuple[HSVTriplet, HSVTriplet]

_ALLOWED_KEYS = {"px_per_meter", "min_area_px", "routes"}


@dataclass(frozen=True)
class RouteSpec:
    """One route colour as seen on the field.

    ``hsv_ranges`` holds one or more ``(lower, upper)`` HSV pairs; red needs two
    ranges because its hue wraps around 180 in OpenCV's H range.
    """

    name: str
    number: int | None
    hsv_ranges: tuple[HSVRange, ...]
    dangerous: bool = False


@dataclass(frozen=True)
class SceneConfig:
    """Everything ``mars.vision`` needs to interpret a frame."""

    routes: tuple[RouteSpec, ...]
    px_per_meter: float | None
    min_area_px: int = 300


def load_config(path: str | Path) -> SceneConfig:
    """Load and validate a scene configuration from ``path``.

    Raises ``FileNotFoundError`` when the file is missing (message names the
    path) and ``ValueError`` with a descriptive message for malformed content.
    """
    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"config file not found: {config_path}") from exc
    except OSError as exc:
        raise OSError(f"cannot read config file {config_path}: {exc}") from exc

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{config_path}: invalid JSON at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(f"{config_path}: top-level JSON value must be an object")

    unknown = sorted(set(data) - _ALLOWED_KEYS)
    if unknown:
        raise ValueError(
            f"{config_path}: unknown key(s) {unknown}; "
            f"allowed keys are {sorted(_ALLOWED_KEYS)}"
        )

    if "routes" not in data:
        raise ValueError(f"{config_path}: missing required key 'routes'")
    raw_routes = data["routes"]
    if not isinstance(raw_routes, list):
        raise ValueError(f"{config_path}: 'routes' must be a list")

    routes = tuple(
        _parse_route(config_path, index, raw)
        for index, raw in enumerate(raw_routes)
    )

    px_per_meter = data.get("px_per_meter")
    if px_per_meter is not None:
        if isinstance(px_per_meter, bool) or not isinstance(px_per_meter, (int, float)):
            raise ValueError(f"{config_path}: 'px_per_meter' must be a number or null")
        if px_per_meter <= 0:
            raise ValueError(f"{config_path}: 'px_per_meter' must be positive")
        px_per_meter = float(px_per_meter)

    min_area_px = data.get("min_area_px", 300)
    if isinstance(min_area_px, bool) or not isinstance(min_area_px, int):
        raise ValueError(f"{config_path}: 'min_area_px' must be an integer")
    if min_area_px < 0:
        raise ValueError(f"{config_path}: 'min_area_px' must be >= 0")

    return SceneConfig(
        routes=routes,
        px_per_meter=px_per_meter,
        min_area_px=min_area_px,
    )


def _parse_route(path: Path, index: int, raw: object) -> RouteSpec:
    where = f"{path}: routes[{index}]"
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be an object")

    unknown = sorted(set(raw) - {"name", "number", "hsv_ranges", "dangerous"})
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown}")

    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{where}: 'name' must be a non-empty string")

    number = raw.get("number")
    if number is not None:
        if isinstance(number, bool) or not isinstance(number, int):
            raise ValueError(f"{where}: 'number' must be an integer or null")

    raw_ranges = raw.get("hsv_ranges")
    if not isinstance(raw_ranges, list) or not raw_ranges:
        raise ValueError(f"{where}: 'hsv_ranges' must be a non-empty list")
    hsv_ranges = tuple(
        _parse_hsv_range(where, range_index, item)
        for range_index, item in enumerate(raw_ranges)
    )

    dangerous = raw.get("dangerous", False)
    if not isinstance(dangerous, bool):
        raise ValueError(f"{where}: 'dangerous' must be a boolean")

    return RouteSpec(
        name=name.strip().lower(),
        number=number,
        hsv_ranges=hsv_ranges,
        dangerous=dangerous,
    )


def _parse_hsv_range(where: str, index: int, raw: object) -> HSVRange:
    label = f"{where}.hsv_ranges[{index}]"
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError(f"{label} must be a pair [lower, upper]")
    lower = _parse_hsv_triplet(f"{label}[0]", raw[0])
    upper = _parse_hsv_triplet(f"{label}[1]", raw[1])
    for channel, (low, high) in enumerate(zip(lower, upper)):
        if low > high:
            raise ValueError(
                f"{label}: lower bound exceeds upper bound in channel {channel}"
            )
    return lower, upper


def _parse_hsv_triplet(label: str, raw: object) -> HSVTriplet:
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError(f"{label} must be a list of three integers [h, s, v]")
    values: list[int] = []
    for channel, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{label}: channel {channel} must be an integer")
        values.append(value)
    hue, saturation, brightness = values
    if not 0 <= hue <= 180:
        raise ValueError(f"{label}: hue must be within 0..180")
    if not 0 <= saturation <= 255:
        raise ValueError(f"{label}: saturation must be within 0..255")
    if not 0 <= brightness <= 255:
        raise ValueError(f"{label}: value must be within 0..255")
    return hue, saturation, brightness
