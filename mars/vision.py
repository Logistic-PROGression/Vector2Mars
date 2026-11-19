"""Colour segmentation and route-length measurement.

The public entry point is :func:`analyze_frame`. It is deliberately defensive:
a bad, empty or missing frame must never crash the flight program, so it
returns an empty :class:`SceneResult` instead of raising.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from mars.config import RouteSpec, SceneConfig

# 5x5 elliptical kernel: removes speckle noise while keeping route strokes.
_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_SKELETON_ELEMENT = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

# Rough BGR rendering colours for the debug overlay. "black" is drawn dark grey
# so the outline stays visible on a dark field.
_ROUTE_BGR: dict[str, tuple[int, int, int]] = {
    "red": (0, 0, 255),
    "green": (0, 200, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "orange": (0, 165, 255),
    "purple": (255, 0, 255),
    "white": (255, 255, 255),
    "black": (80, 80, 80),
}
_FALLBACK_BGR = (255, 255, 255)


@dataclass
class RouteMeasurement:
    """A single detected route."""

    name: str
    number: int | None
    length_px: float
    length_m: float | None
    area_px: int
    dangerous: bool
    # Pixel mask kept for the debug overlay. Excluded from repr/equality so
    # measurements stay cheap to print and compare.
    mask: np.ndarray | None = field(default=None, repr=False, compare=False)


@dataclass
class SceneResult:
    """All routes found in one frame."""

    routes: list[RouteMeasurement] = field(default_factory=list)
    shortest: RouteMeasurement | None = None
    excluded: list[RouteMeasurement] = field(default_factory=list)


def analyze_frame(image_bgr: object, config: SceneConfig) -> SceneResult:
    """Detect every configured route in ``image_bgr`` and pick the shortest safe one.

    Accepts BGR, grayscale or BGRA ``uint8`` arrays and never mutates the input.
    Any unsupported input (``None``, wrong dtype, empty array, unexpected OpenCV
    error) yields an empty result rather than an exception.
    """
    try:
        bgr = _as_bgr(image_bgr)
        if bgr is None or config is None:
            return SceneResult()

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        routes: list[RouteMeasurement] = []
        for spec in config.routes:
            mask, area_px = _segment(hsv, spec, config.min_area_px)
            if area_px <= 0:
                continue
            length_px = skeleton_length_px(mask)
            if length_px <= 0:
                continue
            length_m = (
                None if config.px_per_meter is None else length_px / config.px_per_meter
            )
            routes.append(
                RouteMeasurement(
                    name=spec.name,
                    number=spec.number,
                    length_px=length_px,
                    length_m=length_m,
                    area_px=area_px,
                    dangerous=spec.dangerous,
                    mask=mask,
                )
            )

        excluded = [route for route in routes if route.dangerous]
        candidates = [route for route in routes if not route.dangerous]
        shortest = min(candidates, key=_candidate_key) if candidates else None
        return SceneResult(routes=routes, shortest=shortest, excluded=excluded)
    except Exception:
        # Defensive by design: a malformed frame or config must not stop the
        # drone program. The caller simply sees "no routes".
        return SceneResult()


def annotate(image_bgr: object, result: SceneResult | None) -> np.ndarray:
    """Return a debug copy of ``image_bgr`` with route outlines and labels."""
    base = _as_bgr(image_bgr)
    if base is None:
        base = np.zeros((480, 640, 3), dtype=np.uint8)
    canvas = base.copy()
    if result is None or not result.routes:
        return canvas

    for route in result.routes:
        color = _ROUTE_BGR.get(route.name.lower(), _FALLBACK_BGR)
        thickness = 3 if _same_route(route, result.shortest) else 1
        _draw_route(canvas, route, color, thickness)
    return canvas


def skeleton_length_px(mask: object) -> float:
    """Approximate the length of a binary mask in pixels.

    The mask is thinned with an iterative morphological skeleton
    (``img - open(img)`` accumulated, then erode) followed by a Zhang-Suen pass.
    The morphological skeleton alone leaves a 2-pixel-thick remnant when the
    stroke has an even width, which would double the measured length, so the
    Zhang-Suen pass guarantees a 1-pixel skeleton. The length is then estimated
    as one unit per skeleton pixel plus ``sqrt(2) - 1`` for every diagonal
    8-neighbour step, so diagonal strokes are not undercounted by the pixel
    count. This is an approximation: branch points and end spurs can add a few
    pixels, which is acceptable when only relative route lengths matter.
    """
    skeleton = _skeletonize(mask)
    if skeleton.size == 0:
        return 0.0
    filled = skeleton > 0
    pixel_count = int(np.count_nonzero(filled))
    if pixel_count == 0:
        return 0.0
    diagonal_steps = int(np.count_nonzero(filled[:-1, :-1] & filled[1:, 1:]))
    diagonal_steps += int(np.count_nonzero(filled[:-1, 1:] & filled[1:, :-1]))
    return float(pixel_count) + (math.sqrt(2.0) - 1.0) * diagonal_steps


def _candidate_key(route: RouteMeasurement) -> tuple:
    """Deterministic ordering: length first, then number presence/value, then name."""
    return (
        route.length_px,
        route.number is None,
        route.number if route.number is not None else 0,
        route.name,
    )


def _same_route(left: RouteMeasurement, right: RouteMeasurement | None) -> bool:
    return (
        right is not None
        and left.name == right.name
        and left.number == right.number
    )


def _as_bgr(image: object) -> np.ndarray | None:
    """Return a BGR ``uint8`` copy of a supported image, or ``None``."""
    if image is None:
        return None
    if not isinstance(image, np.ndarray):
        try:
            image = np.asarray(image)
        except Exception:
            return None
    if not isinstance(image, np.ndarray) or image.size == 0:
        return None
    if image.dtype != np.uint8 or image.ndim not in (2, 3):
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    channels = image.shape[2]
    if channels == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    if channels == 3:
        return image.copy()
    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return None


def _segment(
    hsv: np.ndarray, spec: RouteSpec, min_area_px: int
) -> tuple[np.ndarray, int]:
    """Return the filtered mask for one route and its total area in pixels."""
    mask: np.ndarray | None = None
    for lower, upper in spec.hsv_ranges:
        part = cv2.inRange(
            hsv,
            np.asarray(lower, dtype=np.uint8),
            np.asarray(upper, dtype=np.uint8),
        )
        mask = part if mask is None else cv2.bitwise_or(mask, part)

    if mask is None:
        return np.zeros(hsv.shape[:2], dtype=np.uint8), 0

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    kept = [
        label
        for label in range(1, count)
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area_px
    ]
    if not kept:
        return np.zeros_like(mask), 0

    lookup = np.zeros(count, dtype=np.uint8)
    lookup[kept] = 255
    filtered = lookup[labels]
    area_px = sum(int(stats[label, cv2.CC_STAT_AREA]) for label in kept)
    return filtered, area_px


def _skeletonize(mask: object) -> np.ndarray:
    """Iterative morphological skeleton; returns a ``uint8`` 0/255 image."""
    if mask is None:
        return np.zeros((0, 0), dtype=np.uint8)
    image = mask if isinstance(mask, np.ndarray) else np.asarray(mask)
    if image.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim != 2:
        return np.zeros((0, 0), dtype=np.uint8)

    current = np.where(image > 0, 255, 0).astype(np.uint8)
    skeleton = np.zeros_like(current)
    # An erosion always shrinks a finite shape, so this bound is never hit in
    # practice; it only guards against pathological masks.
    max_iterations = int(current.shape[0]) + int(current.shape[1]) + 2
    for _ in range(max_iterations):
        if cv2.countNonZero(current) == 0:
            break
        opened = cv2.morphologyEx(current, cv2.MORPH_OPEN, _SKELETON_ELEMENT)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(current, opened))
        eroded = cv2.erode(current, _SKELETON_ELEMENT)
        if cv2.countNonZero(eroded) == cv2.countNonZero(current):
            break
        current = eroded
    return _thin(skeleton)


def _thin(binary: np.ndarray) -> np.ndarray:
    """Reduce a binary mask to a 1-pixel skeleton with Zhang-Suen thinning.

    Vectorised numpy implementation; the loop ends when a full pass removes
    nothing. Works without ``cv2.ximgproc``.
    """
    image = np.where(binary > 0, 1, 0).astype(np.uint8)
    if image.size == 0:
        return np.zeros_like(binary)

    while True:
        changed = False
        for step in (0, 1):
            padded = np.pad(image, 1, mode="constant")
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]

            neighbours = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
                + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
                + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
                + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
                + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
                + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
                + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
            )

            base = (
                (image == 1)
                & (neighbours >= 2)
                & (neighbours <= 6)
                & (transitions == 1)
            )
            if step == 0:
                removable = base & ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
            else:
                removable = base & ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)

            if removable.any():
                image[removable] = 0
                changed = True
        if not changed:
            break
    return (image * 255).astype(np.uint8)


def _draw_route(
    canvas: np.ndarray,
    route: RouteMeasurement,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    mask = route.mask
    if mask is None or mask.size == 0:
        return
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return
    cv2.drawContours(canvas, contours, -1, color, thickness)

    moments = cv2.moments(mask, binaryImage=True)
    if moments["m00"] <= 0:
        return
    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])

    label = route.name
    if route.number is not None:
        label += f" #{route.number}"
    label += f" {route.length_px:.0f}px"
    if route.length_m is not None:
        label += f" ({route.length_m:.2f}m)"
    text_color = color if sum(color) >= 120 else (255, 255, 255)
    cv2.putText(
        canvas,
        label,
        (center_x + 6, center_y - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        text_color,
        1,
        cv2.LINE_AA,
    )
