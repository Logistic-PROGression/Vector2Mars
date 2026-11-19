"""Detect arrow-marked routes between ArUco markers.

The competition map encodes routes as chains of printed arrows. Chains can be
curved, arrows can be any size and any of several colours (white, red, blue),
so the detector works on shape rather than on a fixed colour or a straight
corridor:

1. arrow candidates are closed printed silhouettes found on the Canny edge
   map, plus blobs of saturated colour (red/blue) on the colour masks;
2. every candidate is filtered by size/solidity expressed in marker sides, so
   the same code works on a 347x446 satellite image and on a 1920x2560 frame;
3. candidates are linked into chains by a sparse neighbour graph (distance,
   axis agreement), shortcut edges are pruned, and each chain is ordered;
4. a chain becomes a route only when both of its ends sit next to a marker,
   so a route is always identified by its marker pair. Several routes and
   several markers in one frame therefore stay separate.

The module is defensive: a missing, empty or malformed frame yields ``[]``.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from mars.aruco import DEFAULT_DICTIONARY, Marker, annotate_markers, detect_markers

# All geometric limits are in marker sides ("unit"), never in pixels.
MIN_AREA_FRACTION = 0.03
MAX_AREA_FRACTION = 1.0
MIN_SIDE_FRACTION = 0.2
MAX_SIDE_FRACTION = 1.8
MAX_BBOX_ASPECT = 2.8
MIN_SOLIDITY = 0.45  # thin shafts of long arrows fill only ~half of their hull
MIN_AREA_PX = 25.0

# Chain graph: arrows of one route are close and point roughly along the path.
LINK_MAX_STEP_UNITS = 2.6
LINK_MAX_TURN_DEG = 50.0
LINK_NEIGHBOURS = 2
PRUNE_DETOUR_RATIO = 1.12
MARKER_REACH_UNITS = 4.0

COLOUR_RED = "red"
COLOUR_BLUE = "blue"
COLOUR_WHITE = "white"
COLOUR_LABELS_RU = {COLOUR_RED: "красных", COLOUR_BLUE: "синих", COLOUR_WHITE: "белых"}


@dataclass(frozen=True)
class Arrow:
    """One arrow of a route, oriented from the route start to its end."""

    center: tuple[float, float]
    direction: tuple[float, float]  # unit vector the arrowhead points to
    area_px: int
    length_px: float
    color: str = COLOUR_WHITE


@dataclass(frozen=True)
class ArrowRoute:
    """An ordered arrow chain connecting two ArUco markers."""

    start_marker_id: int
    end_marker_id: int
    arrows: tuple[Arrow, ...]
    polyline: tuple[tuple[float, float], ...]
    length_px: float
    length_m: float | None = None

    @property
    def colors(self) -> tuple[str, ...]:
        """Colours of the arrows in route order."""
        return tuple(arrow.color for arrow in self.arrows)


@dataclass
class _Arrow:
    """Internal arrow candidate.

    ``head_sign`` is +1 when the wide end of the arrow lies towards ``axis``.
    """

    index: int
    center: np.ndarray
    axis: np.ndarray
    head_sign: int
    area_px: int
    length_px: float
    color: str = COLOUR_WHITE


@dataclass
class _Chain:
    """A verified route candidate: two markers and the arrows between them."""

    start: Marker
    end: Marker
    arrows: list[_Arrow] = field(default_factory=list)


def detect_arrow_routes(
    image: object,
    markers: list[Marker] | None = None,
    *,
    dictionary: str = DEFAULT_DICTIONARY,
    px_per_meter: float | None = None,
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

    arrows = _find_arrows(bgr, marker_list, unit)
    if len(arrows) < min_arrows:
        return []

    adjacency = _link_graph(arrows, unit)
    adjacency = _prune_shortcuts(adjacency, arrows)

    chains: list[_Chain] = []
    for component in _components(adjacency):
        if len(component) < min_arrows:
            continue
        path = _order_chain(adjacency, component, arrows)
        if len(path) < min_arrows:
            continue
        chain = _match_markers(path, arrows, marker_list, unit)
        if chain is not None:
            chains.append(chain)

    # One route per marker pair: keep the chain with more arrows.
    chains.sort(key=lambda chain: (-len(chain.arrows), chain.start.id, chain.end.id))
    used_pairs: set[frozenset[int]] = set()
    used_arrows: set[int] = set()
    routes: list[ArrowRoute] = []
    for chain in chains:
        pair = frozenset((chain.start.id, chain.end.id))
        indices = {arrow.index for arrow in chain.arrows}
        if pair in used_pairs or used_arrows.intersection(indices):
            continue
        used_pairs.add(pair)
        used_arrows.update(indices)
        routes.append(_to_public(chain, px_per_meter, unit))
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
            cv2.circle(canvas, center, 4, color, -1)

        length = (
            f"{route.length_m:.2f} м"
            if route.length_m is not None
            else f"{route.length_px:.0f} px"
        )
        label = (
            f"{route.start_marker_id}->{route.end_marker_id}: "
            f"стрелок {len(route.arrows)}, {length}"
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


def _find_arrows(bgr: np.ndarray, markers: list[Marker], unit: float) -> list[_Arrow]:
    """Return printed arrow silhouettes outside the marker quiet zones."""
    gray = cv2.bilateralFilter(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 7, 40, 40)
    height, width = gray.shape[:2]

    # Printed arrows always have a thin dark outline, so the edge map gives a
    # closed silhouette whatever colour the arrow itself is.
    edges = cv2.Canny(gray, 25, 70)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # Saturated arrows (red/blue) also give solid colour blobs; adding them
    # recovers arrows whose outline is broken by glare.
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red_mask = (((hue <= 14) | (hue >= 163)) & (saturation > 110) & (value > 60))
    blue_mask = ((hue >= 95) & (hue <= 140) & (saturation > 80) & (value > 45))
    sources = [edges]
    for mask in (red_mask, blue_mask):
        filled = (mask.astype(np.uint8) * 255)
        sources.append(cv2.morphologyEx(filled, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)))

    min_area = max(MIN_AREA_FRACTION * unit * unit, MIN_AREA_PX)
    max_area = MAX_AREA_FRACTION * unit * unit
    min_side = MIN_SIDE_FRACTION * unit
    max_side = MAX_SIDE_FRACTION * unit
    margin_x = 0.01 * width
    margin_y = 0.01 * height

    candidates: list[tuple[np.ndarray, int, str, np.ndarray]] = []
    for source in sources:
        contours, _ = cv2.findContours(source, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if not min_area <= area <= max_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if not (min_side <= w <= max_side and min_side <= h <= max_side):
                continue
            if max(w, h) / max(min(w, h), 1) > MAX_BBOX_ASPECT:
                continue
            hull = cv2.convexHull(contour)
            if area / max(cv2.contourArea(hull), 1.0) < MIN_SOLIDITY:
                continue
            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            if not 3 <= len(polygon) <= 7:
                continue
            if x < margin_x or y < margin_y:
                continue
            if x + w > width - margin_x or y + h > height - margin_y:
                continue

            center = np.asarray((x + w / 2.0, y + h / 2.0), dtype=np.float64)
            if any(
                np.linalg.norm(center - np.asarray(marker.center)) < 1.0 * marker.side_px
                for marker in markers
            ):
                continue

            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
            candidates.append((center, int(area), _color_of(bgr, mask), contour))

    # Nested contours of one arrow (outer + inner edge) collapse to the largest.
    candidates.sort(key=lambda item: -item[1])
    unique: list[tuple[np.ndarray, int, str, np.ndarray]] = []
    for candidate in candidates:
        center = candidate[0]
        if any(np.linalg.norm(center - kept[0]) < 0.3 * unit for kept in unique):
            continue
        unique.append(candidate)

    result: list[_Arrow] = []
    for center, area, color, contour in unique:
        axis, head_sign, length = _axis_and_head(contour)
        if axis is None:
            continue
        result.append(
            _Arrow(
                index=len(result),
                center=center,
                axis=axis,
                head_sign=head_sign,
                area_px=area,
                length_px=length,
                color=color,
            )
        )
    return result


def _axis_and_head(contour: np.ndarray) -> tuple[np.ndarray | None, int, float]:
    """Return the arrow axis, the side of the head (+1/-1) and its length."""
    points = contour.reshape(-1, 2).astype(np.float32)
    if len(points) < 4:
        return None, 0, 0.0
    mean = points.mean(axis=0)
    covariance = np.cov((points - mean).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues[-1] <= 0:
        return None, 0, 0.0
    axis = eigenvectors[:, -1]
    if axis[0] < 0 or (abs(axis[0]) < 1e-6 and axis[1] < 0):
        axis = -axis

    projection = (points - mean) @ axis
    perpendicular = (points - mean) @ np.asarray((-axis[1], axis[0]))
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
    head_sign = 0
    if abs(head - tail) >= 0.5:
        head_sign = 1 if head > tail else -1
    return axis, head_sign, float(projection.max() - projection.min())


def _color_of(bgr: np.ndarray, mask: np.ndarray) -> str:
    """Classify the arrow colour by mean channel values inside ``mask``."""
    blue, green, red = (float(channel) for channel in cv2.mean(bgr, mask=mask)[:3])
    if red > 1.35 * max(green, blue) and red > 70:
        return COLOUR_RED
    if blue > 1.15 * max(red, green) and blue > 60:
        return COLOUR_BLUE
    return COLOUR_WHITE


def _link_graph(arrows: list[_Arrow], unit: float) -> dict[int, set[int]]:
    """Connect each arrow to its nearest compatible neighbours."""
    adjacency: dict[int, set[int]] = {arrow.index: set() for arrow in arrows}
    cos_turn = math.cos(math.radians(LINK_MAX_TURN_DEG))
    for first in arrows:
        candidates: list[tuple[float, int]] = []
        for second in arrows:
            if first.index == second.index:
                continue
            delta = second.center - first.center
            distance = float(np.linalg.norm(delta))
            if not 0.4 * unit <= distance <= LINK_MAX_STEP_UNITS * unit:
                continue
            direction = delta / distance
            if abs(float(direction @ first.axis)) < cos_turn:
                continue
            if abs(float(direction @ second.axis)) < cos_turn:
                continue
            candidates.append((distance, second.index))
        candidates.sort()
        for _, index in candidates[:LINK_NEIGHBOURS]:
            adjacency[first.index].add(index)
            adjacency[index].add(first.index)
    return adjacency


def _prune_shortcuts(
    adjacency: dict[int, set[int]], arrows: list[_Arrow]
) -> dict[int, set[int]]:
    """Drop edges that jump over a nearer arrow of the same chain."""
    by_index = {arrow.index: arrow for arrow in arrows}
    pruned = {index: set(neighbours) for index, neighbours in adjacency.items()}
    for first, neighbours in adjacency.items():
        for second in list(neighbours):
            if second <= first:
                continue
            direct = float(np.linalg.norm(by_index[first].center - by_index[second].center))
            for via in adjacency[first] & adjacency[second]:
                first_leg = float(np.linalg.norm(by_index[first].center - by_index[via].center))
                second_leg = float(np.linalg.norm(by_index[via].center - by_index[second].center))
                if first_leg + second_leg <= direct * PRUNE_DETOUR_RATIO:
                    pruned[first].discard(second)
                    pruned[second].discard(first)
                    break
    return pruned


def _components(adjacency: dict[int, set[int]]) -> list[list[int]]:
    """Split the graph into connected components."""
    seen: set[int] = set()
    components: list[list[int]] = []
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbour in adjacency[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        components.append(sorted(component))
    return components


def _order_chain(
    adjacency: dict[int, set[int]], component: list[int], arrows: list[_Arrow]
) -> list[int]:
    """Order one component into a chain following direction continuity."""
    by_index = {arrow.index: arrow for arrow in arrows}
    nodes = set(component)
    if len(nodes) == 1:
        return list(nodes)

    ends = [node for node in nodes if len(adjacency[node] & nodes) <= 1]
    start = min(ends) if ends else component[0]
    path = [start]
    previous, current = None, start
    while True:
        options = [
            node
            for node in adjacency[current] & nodes
            if node != previous and node not in path
        ]
        if not options:
            break
        if previous is None:
            options.sort(key=lambda node: -float(np.linalg.norm(by_index[node].center - by_index[current].center)))
            following = options[0]
        else:
            heading = by_index[current].center - by_index[previous].center
            heading /= max(float(np.linalg.norm(heading)), 1e-9)
            following = max(
                options,
                key=lambda node: _direction(by_index[current], by_index[node]) @ heading,
            )
        previous, current = current, following
        path.append(current)
    return path


def _direction(origin: _Arrow, target: _Arrow) -> np.ndarray:
    """Unit vector from ``origin`` to ``target``."""
    delta = target.center - origin.center
    return delta / max(float(np.linalg.norm(delta)), 1e-9)


def _match_markers(
    path: list[int], arrows: list[_Arrow], markers: list[Marker], unit: float
) -> _Chain | None:
    """Attach chain ends to the closest pair of different markers."""
    by_index = {arrow.index: arrow for arrow in arrows}
    head = by_index[path[0]]
    tail = by_index[path[-1]]
    best: tuple[float, Marker, Marker] | None = None
    for start in markers:
        for end in markers:
            if start.id == end.id:
                continue
            head_gap = float(np.linalg.norm(head.center - np.asarray(start.center)))
            tail_gap = float(np.linalg.norm(tail.center - np.asarray(end.center)))
            if head_gap > MARKER_REACH_UNITS * unit or tail_gap > MARKER_REACH_UNITS * unit:
                continue
            if best is None or head_gap + tail_gap < best[0]:
                best = (head_gap + tail_gap, start, end)
    if best is None:
        return None
    _, start, end = best
    return _Chain(start=start, end=end, arrows=[by_index[index] for index in path])


def _to_public(chain: _Chain, px_per_meter: float | None, unit: float) -> ArrowRoute:
    """Convert an internal chain into a public route with a metric length."""
    arrows = list(chain.arrows)
    start_center = np.asarray(chain.start.center, dtype=np.float64)
    end_center = np.asarray(chain.end.center, dtype=np.float64)

    # Majority vote of arrowheads decides the travel direction.
    votes = 0
    for index, arrow in enumerate(arrows):
        if index + 1 < len(arrows):
            travel = arrows[index + 1].center - arrow.center
        else:
            travel = arrow.center - arrows[index - 1].center
            travel = -travel
        travel /= max(float(np.linalg.norm(travel)), 1e-9)
        votes += arrow.head_sign * (1 if float(arrow.axis @ travel) >= 0 else -1)

    if votes < 0:
        arrows.reverse()
        start_center, end_center = end_center, start_center
        chain = _Chain(start=chain.end, end=chain.start, arrows=arrows)

    route_direction = end_center - start_center
    norm = float(np.linalg.norm(route_direction))
    route_direction = route_direction / norm if norm > 0 else np.asarray((1.0, 0.0))

    public_arrows: list[Arrow] = []
    for arrow in arrows:
        axis = arrow.axis if float(arrow.axis @ route_direction) >= 0 else -arrow.axis
        public_arrows.append(
            Arrow(
                center=(float(arrow.center[0]), float(arrow.center[1])),
                direction=(float(axis[0]), float(axis[1])),
                area_px=arrow.area_px,
                length_px=arrow.length_px,
                color=arrow.color,
            )
        )

    points = [tuple(chain.start.center)]
    points.extend(arrow.center for arrow in public_arrows)
    points.append(tuple(chain.end.center))
    polyline = tuple((float(x), float(y)) for x, y in points)
    length_px = sum(
        math.dist(polyline[index], polyline[index + 1])
        for index in range(len(polyline) - 1)
    )
    length_m = None if px_per_meter is None else length_px / px_per_meter
    return ArrowRoute(
        start_marker_id=chain.start.id,
        end_marker_id=chain.end.id,
        arrows=tuple(public_arrows),
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
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
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
        mixed = Counter(route.colors)
        breakdown = ", ".join(
            f"{COLOUR_LABELS_RU[color]} {mixed[color]}"
            for color in (COLOUR_RED, COLOUR_BLUE, COLOUR_WHITE)
            if mixed[color]
        )
        print(
            f"  {route.start_marker_id} -> {route.end_marker_id}: "
            f"стрелок {len(route.arrows)} ({breakdown}), длина {length}"
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
