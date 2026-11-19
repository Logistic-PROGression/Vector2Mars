"""Detect arrow-marked routes between ArUco markers.

The satellite photo encodes the dangerous route as a chain of white arrows
between two markers (task 2, step B4). A route is therefore identified by its
marker pair: for every pair of detected markers the module projects bright
arrow-shaped blobs onto the marker-to-marker axis, orders them, checks spacing
and direction, and returns one :class:`ArrowRoute`. Keying routes by marker ids
keeps several routes and markers in one frame separate.

The module is defensive: a missing or malformed frame yields an empty list.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from mars.aruco import DEFAULT_DICTIONARY, Marker, annotate_markers, detect_markers

# White/threshold defaults tuned on samples/image.jpg (arrows are much brighter
# than the floor). Recalibrate on real frames if the lighting changes.
WHITE_V_MIN = 160
WHITE_S_MAX = 70

# All geometric limits are expressed in marker sides ("unit") so the detector
# works both on the 347x446 satellite image and on high-resolution camera frames.
MIN_AREA_FRACTION = 0.03
MAX_AREA_FRACTION = 8.0
MIN_SIDE_FRACTION = 0.2
MAX_SIDE_FRACTION = 3.0
MAX_BBOX_ASPECT = 4.0
MAX_ELONGATION = 5.0
MAX_GAP_UNITS = 3.0
MAX_MARKER_DISTANCE_UNITS = 4.0
MAX_FIT_RESIDUAL_UNITS = 1.0
MIN_SPAN_FRACTION = 0.3
ALIGNMENT_DEG = 45.0


@dataclass(frozen=True)
class Arrow:
    """One arrow of a route, oriented from the route start to its end."""

    center: tuple[float, float]
    direction: tuple[float, float]  # unit vector the arrowhead points to
    area_px: int
    length_px: float


@dataclass(frozen=True)
class ArrowRoute:
    """An ordered arrow chain connecting two ArUco markers."""

    start_marker_id: int
    end_marker_id: int
    arrows: tuple[Arrow, ...]
    polyline: tuple[tuple[float, float], ...]
    length_px: float
    length_m: float | None = None


@dataclass(frozen=True)
class _Arrow:
    """Internal candidate: sign of ``head_sign`` is relative to ``axis``."""

    index: int
    center: tuple[float, float]
    axis: tuple[float, float]
    head_sign: int
    area_px: int
    length_px: float


@dataclass(frozen=True)
class _RouteCandidate:
    start: Marker
    end: Marker
    arrows: tuple[_Arrow, ...]
    score: float
    indices: frozenset[int] = field(default=frozenset())


def detect_arrow_routes(
    image: object,
    markers: list[Marker] | None = None,
    *,
    dictionary: str = DEFAULT_DICTIONARY,
    px_per_meter: float | None = None,
    white_v_min: int = WHITE_V_MIN,
    white_s_max: int = WHITE_S_MAX,
    min_arrows: int = 2,
) -> list[ArrowRoute]:
    """Find arrow chains that connect pairs of ArUco markers.

    ``markers`` can be passed by the caller (recommended: detect once and reuse)
    or is detected with :func:`mars.aruco.detect_markers`. Routes need at least
    two markers, so an image without markers yields an empty list.
    """
    bgr = _as_bgr(image)
    if bgr is None:
        return []

    if markers is None:
        markers = detect_markers(bgr, dictionary)
    marker_list = sorted(markers or (), key=lambda marker: marker.id)
    if len(marker_list) < 2:
        return []

    unit = float(np.median([marker.side_px for marker in marker_list]))
    if not math.isfinite(unit) or unit <= 0:
        return []

    arrows = _find_arrows(bgr, marker_list, unit, white_v_min, white_s_max)
    if len(arrows) < min_arrows:
        return []

    candidates: list[_RouteCandidate] = []
    for first, second in itertools.combinations(marker_list, 2):
        route = _route_for_pair(first, second, arrows, unit, min_arrows)
        if route is not None:
            candidates.append(route)

    # Prefer routes with more arrows, then better geometric score. Greedy
    # assignment ensures one arrow belongs to at most one route.
    candidates.sort(key=lambda route: (-len(route.arrows), -route.score))
    used: set[int] = set()
    routes: list[ArrowRoute] = []
    for candidate in candidates:
        if used.intersection(candidate.indices):
            continue
        used.update(candidate.indices)
        routes.append(_to_public(candidate, px_per_meter))
    return routes


def annotate_arrow_routes(
    image: object, routes: list[ArrowRoute] | None, markers: list[Marker] | None
) -> np.ndarray:
    """Return a debug copy with arrow routes highlighted and markers on top."""
    base = _as_bgr(image)
    if base is None:
        base = np.zeros((480, 640, 3), dtype=np.uint8)
    canvas = base.copy()
    colors = [
        (0, 165, 255),
        (255, 0, 255),
        (255, 255, 0),
        (0, 0, 255),
        (180, 105, 255),
    ]
    for index, route in enumerate(routes or ()):
        color = colors[index % len(colors)]
        points = np.asarray(route.polyline, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], False, color, 3, cv2.LINE_AA)
        for arrow in route.arrows:
            center = (int(round(arrow.center[0])), int(round(arrow.center[1])))
            cv2.circle(canvas, center, 3, color, -1)

        label = (
            f"{route.start_marker_id}->{route.end_marker_id}: "
            f"стрелок {len(route.arrows)}, {route.length_px:.0f} px"
        )
        if route.arrows:
            anchor = np.mean(
                np.asarray([arrow.center for arrow in route.arrows]), axis=0
            )
            origin = (int(anchor[0]) + 8, int(anchor[1]) - 12)
        else:
            origin = (int(points[0][0][0]) + 6, int(points[0][0][1]) - 6)
        _put_label(canvas, label, origin, color)

    # Draw markers last so their outlines and ids stay visible above routes.
    return annotate_markers(canvas, markers)
    return canvas


def _find_arrows(
    bgr: np.ndarray,
    markers: list[Marker],
    unit: float,
    white_v_min: int,
    white_s_max: int,
) -> list[_Arrow]:
    """Return bright, arrow-shaped components outside the marker quiet zones."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.asarray((0, 0, white_v_min), dtype=np.uint8),
        np.asarray((180, white_s_max, 255), dtype=np.uint8),
    )

    # Marker cells and their white borders must not count as arrows.
    exclusion = np.zeros(mask.shape[:2], dtype=np.uint8)
    for marker in markers:
        cv2.fillConvexPoly(exclusion, marker.corners.astype(np.int32), 255)
        center = (int(round(marker.center[0])), int(round(marker.center[1])))
        cv2.circle(exclusion, center, int(round(unit * 0.9)), 255, -1)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(exclusion))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    height, width = mask.shape[:2]
    min_area = MIN_AREA_FRACTION * unit * unit
    max_area = MAX_AREA_FRACTION * unit * unit
    min_side = MIN_SIDE_FRACTION * unit
    max_side = MAX_SIDE_FRACTION * unit
    cos_alignment = math.cos(math.radians(ALIGNMENT_DEG))

    arrows: list[_Arrow] = []
    for label in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[label])
        if not min_area <= area <= max_area:
            continue
        if not (min_side <= w <= max_side and min_side <= h <= max_side):
            continue
        if max(w, h) / max(min(w, h), 1) > MAX_BBOX_ASPECT:
            continue
        if x <= 0 or y <= 0 or x + w >= width or y + h >= height:
            continue

        ys, xs = np.where(labels == label)
        points = np.column_stack((xs, ys)).astype(np.float32)
        center = points.mean(axis=0)
        covariance = np.cov((points - center).T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        major, minor = float(eigenvalues[-1]), float(eigenvalues[0])
        if major <= 0 or minor <= 0:
            continue
        if math.sqrt(major / minor) > MAX_ELONGATION:
            continue

        axis = eigenvectors[:, -1]
        if axis[0] < 0 or (abs(axis[0]) < 1e-6 and axis[1] < 0):
            axis = -axis

        projection = (points - center) @ axis
        perpendicular = (points - center) @ np.asarray((-axis[1], axis[0]))
        arrows.append(
            _Arrow(
                index=len(arrows),
                center=(float(center[0]), float(center[1])),
                axis=(float(axis[0]), float(axis[1])),
                head_sign=_head_sign(projection, perpendicular),
                area_px=area,
                length_px=float(projection.max() - projection.min()),
            )
        )
    return arrows


def _head_sign(projection: np.ndarray, perpendicular: np.ndarray) -> int:
    """Return +1 if the arrowhead is at +axis, -1 at -axis, 0 if unclear."""
    if projection.size < 4:
        return 0
    edges = np.linspace(float(projection.min()), float(projection.max()), 5)
    widths: list[float] = []
    for start, end in zip(edges[:-1], edges[1:]):
        selected = (projection >= start) & (projection <= end)
        if selected.any():
            values = perpendicular[selected]
            widths.append(float(values.max() - values.min()))
        else:
            widths.append(0.0)
    tail = widths[0] + widths[1]
    head = widths[-2] + widths[-1]
    if abs(head - tail) < 0.5:  # too flat at low resolution
        return 0
    return 1 if head > tail else -1


def _route_for_pair(
    first: Marker,
    second: Marker,
    arrows: list[_Arrow],
    unit: float,
    min_arrows: int,
) -> _RouteCandidate | None:
    """Collect arrows in the corridor between two markers and order the chain."""
    start = np.asarray(first.center, dtype=np.float64)
    end = np.asarray(second.center, dtype=np.float64)
    delta = end - start
    distance = float(np.linalg.norm(delta))
    if distance <= 0:
        return None
    direction = delta / distance
    normal = np.asarray((-direction[1], direction[0]))

    corridor = min(3.0 * unit, max(1.2 * unit, 0.15 * distance))
    max_gap = MAX_GAP_UNITS * unit
    cos_alignment = math.cos(math.radians(ALIGNMENT_DEG))

    picks: list[tuple[float, float, float, _Arrow]] = []
    for arrow in arrows:
        center = np.asarray(arrow.center, dtype=np.float64)
        relative = center - start
        along = float(relative @ direction)
        across = abs(float(relative @ normal))
        if not -0.5 * unit <= along <= distance + 0.5 * unit:
            continue
        if across > corridor:
            continue
        axis = np.asarray(arrow.axis, dtype=np.float64)
        alignment = abs(float(axis @ direction))
        if alignment < cos_alignment:
            continue
        picks.append((along, across, alignment, arrow))

    if len(picks) < min_arrows:
        return None
    picks.sort(key=lambda item: item[0])

    # Split at suspiciously large gaps and keep the longest contiguous run.
    groups: list[list[tuple[float, float, float, _Arrow]]] = [[picks[0]]]
    for pick in picks[1:]:
        if pick[0] - groups[-1][-1][0] > max_gap:
            groups.append([pick])
        else:
            groups[-1].append(pick)
    group = max(groups, key=lambda items: (len(items), items[-1][0] - items[0][0]))
    if len(group) < min_arrows:
        return None

    span = group[-1][0] - group[0][0]
    if span < MIN_SPAN_FRACTION * distance:
        return None

    # Reject cross-marker chains: arrows of one route must lie on one line.
    centers = np.asarray([item[3].center for item in group], dtype=np.float64)
    centered = centers - centers.mean(axis=0)
    _, _, eigenvectors = np.linalg.svd(centered, full_matrices=False)
    perpendicular_axis = eigenvectors[-1]
    residuals = centered @ perpendicular_axis
    if float(np.sqrt(np.mean(residuals ** 2))) > MAX_FIT_RESIDUAL_UNITS * unit:
        return None

    first_arrow = np.asarray(group[0][3].center, dtype=np.float64)
    last_arrow = np.asarray(group[-1][3].center, dtype=np.float64)
    if np.linalg.norm(first_arrow - start) > MAX_MARKER_DISTANCE_UNITS * unit:
        return None
    if np.linalg.norm(last_arrow - end) > MAX_MARKER_DISTANCE_UNITS * unit:
        return None

    # Arrows point along their axis or opposite it. Majority vote of the head
    # side decides whether the route goes first -> second or the other way.
    votes = 0
    for _, _, _, arrow in group:
        axis = np.asarray(arrow.axis, dtype=np.float64)
        sign = 1 if float(axis @ direction) >= 0 else -1
        votes += arrow.head_sign * sign

    if votes > 0:
        ordered = [item[3] for item in group]
        start_marker, end_marker = first, second
    elif votes < 0:
        ordered = [item[3] for item in reversed(group)]
        start_marker, end_marker = second, first
    else:
        ordered = [item[3] for item in group]
        if first.id <= second.id:
            start_marker, end_marker = first, second
        else:
            ordered.reverse()
            start_marker, end_marker = second, first

    mean_across = sum(item[1] for item in group) / len(group)
    mean_alignment = sum(item[2] for item in group) / len(group)
    score = (
        len(group) * 10.0
        - mean_across / unit
        - 5.0 * (1.0 - mean_alignment)
        + 2.0 * span / distance
        - (
            np.linalg.norm(first_arrow - start) + np.linalg.norm(last_arrow - end)
        )
        / (2.0 * unit)
    )
    return _RouteCandidate(
        start=start_marker,
        end=end_marker,
        arrows=tuple(ordered),
        score=score,
        indices=frozenset(arrow.index for arrow in ordered),
    )


def _to_public(candidate: _RouteCandidate, px_per_meter: float | None) -> ArrowRoute:
    start = np.asarray(candidate.start.center, dtype=np.float64)
    end = np.asarray(candidate.end.center, dtype=np.float64)
    route_direction = end - start
    route_length = float(np.linalg.norm(route_direction))
    route_direction = (
        route_direction / route_length
        if route_length > 0
        else np.asarray((1.0, 0.0))
    )

    arrows: list[Arrow] = []
    for arrow in candidate.arrows:
        axis = np.asarray(arrow.axis, dtype=np.float64)
        if float(axis @ route_direction) < 0:
            axis = -axis
        arrows.append(
            Arrow(
                center=arrow.center,
                direction=(float(axis[0]), float(axis[1])),
                area_px=arrow.area_px,
                length_px=arrow.length_px,
            )
        )

    points = [tuple(candidate.start.center)]
    points.extend(arrow.center for arrow in arrows)
    points.append(tuple(candidate.end.center))
    polyline = tuple((float(x), float(y)) for x, y in points)
    length_px = sum(
        math.dist(polyline[index], polyline[index + 1])
        for index in range(len(polyline) - 1)
    )
    length_m = None if px_per_meter is None else length_px / px_per_meter
    return ArrowRoute(
        start_marker_id=candidate.start.id,
        end_marker_id=candidate.end.id,
        arrows=tuple(arrows),
        polyline=polyline,
        length_px=length_px,
        length_m=length_m,
    )


def _put_label(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    """Draw a route caption with a dark outline, kept inside the canvas."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.5, 1
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, scale, thickness
    )
    max_x = max(0, canvas.shape[1] - text_width - 4)
    max_y = max(text_height + 4, canvas.shape[0] - baseline - 4)
    x = min(max(4, origin[0]), max_x)
    y = min(max(text_height + 4, origin[1]), max_y)
    cv2.putText(canvas, text, (x, y), font, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


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


def main(argv: list[str] | None = None) -> int:
    """CLI: detect arrow routes on an image and optionally save an overlay."""
    parser = argparse.ArgumentParser(
        description="Найти маршруты из стрелок между ArUco-метками."
    )
    parser.add_argument("image", help="путь к изображению (JPEG/PNG)")
    parser.add_argument(
        "--dict",
        default=DEFAULT_DICTIONARY,
        dest="dictionary",
        help=f"словарь OpenCV (по умолчанию {DEFAULT_DICTIONARY})",
    )
    parser.add_argument(
        "--marker-size-m",
        type=float,
        default=None,
        help="физический размер стороны маркера в метрах",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="сохранить изображение с разметкой по указанному пути",
    )
    args = parser.parse_args(argv)

    try:
        image = cv2.imread(str(args.image))
    except cv2.error:
        image = None
    if image is None:
        print(f"Не удалось прочитать изображение: {args.image}", file=sys.stderr)
        return 2

    try:
        markers = detect_markers(image, args.dictionary)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    scale = None
    if args.marker_size_m is not None:
        from mars.scale import px_per_meter

        try:
            scale = px_per_meter(markers, args.marker_size_m)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2

    routes = detect_arrow_routes(image, markers, px_per_meter=scale)
    marker_ids = ", ".join(f"id={marker.id}" for marker in markers)
    print(f"Метки: {len(markers)}" + (f" ({marker_ids})" if markers else ""))
    print(f"Маршруты: {len(routes)}")
    for route in routes:
        length = f"{route.length_px:.1f} px"
        if route.length_m is not None:
            length += f" ({route.length_m:.2f} м)"
        print(
            f"  {route.start_marker_id} -> {route.end_marker_id}: "
            f"стрелок {len(route.arrows)}, длина {length}"
        )

    if args.out:
        path = Path(args.out)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            overlay = annotate_arrow_routes(image, routes, markers)
            if not cv2.imwrite(str(path), overlay):
                print(f"Не удалось сохранить разметку: {path}", file=sys.stderr)
                return 2
        except OSError as exc:
            print(f"Не удалось сохранить разметку: {path}: {exc}", file=sys.stderr)
            return 2
        print(f"Разметка сохранена: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
