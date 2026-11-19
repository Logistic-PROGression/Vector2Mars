"""Detect arrow-marked routes between ArUco markers.

The competition map encodes routes as chains of printed arrows. Chains can be
curved, arrows can be any size and any of several colours (white, red, blue),
so the detector works on shape rather than on a fixed colour or a straight
corridor:

1. arrow candidates are closed printed silhouettes found on the Canny edge
   map, blobs of saturated colour (red/blue) on the colour masks, and blobs of
   white paint left by a local brightness threshold. White paint is only
   accepted when its rim follows a printed outline and it is brighter than the
   floor, so a cast shadow or a bright patch is not an arrow;
2. every candidate is filtered by size/solidity expressed in marker sides, so
   the same code works on a 347x446 satellite image and on a 1920x2560 frame;
3. candidates are linked into chains by a sparse neighbour graph (distance,
   axis agreement), shortcut edges are pruned, and each chain is ordered. A
   link never crosses a marker's quiet zone, because that empty ring is the
   boundary between two routes that terminate at the same marker;
4. a chain becomes a route only when each of its ends sits next to the marker
   nearest to it, so a route is always identified by its marker pair. A chain
   cut by the frame edge therefore stays unreported instead of being attached
   to an unrelated marker or to a marker ring that was cut in two.

The module is defensive: a missing, empty or malformed frame yields ``[]``.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from collections.abc import Sequence
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

# Edge and colour thresholds follow the frame statistics so exposure changes do
# not hide paint: limits are percentiles of this frame, scaled and clamped.
CANNY_PERCENTILE = 99.0
CANNY_SCALE = 0.75
CANNY_HIGH_MIN = 25.0
CANNY_HIGH_MAX = 90.0
CANNY_LOW_RATIO = 0.4
SATURATION_PERCENTILE = 99.0
VALUE_PERCENTILE = 95.0
RED_SATURATION_SCALE = 0.55
RED_SATURATION_MIN = 40.0
RED_SATURATION_MAX = 110.0
BLUE_SATURATION_SCALE = 0.32
BLUE_SATURATION_MIN = 35.0
BLUE_SATURATION_MAX = 90.0
VALUE_MIN_SCALE = 0.25
VALUE_MIN_FLOOR = 30.0
VALUE_MIN_CAP = 60.0

# White paint has no hue to key on: it is the brightest, least saturated thing
# on the floor. The threshold is local, so shadows and glare do not hide it.
WHITE_SATURATION_MAX = 40.0
WHITE_ADAPTIVE_BLOCK_UNITS = 2.0
WHITE_ADAPTIVE_C = 12.0
# Long thin bright seams and tape lines are not arrows: a vertical opening
# drops them before they can merge with an arrow that lies across them. Only
# full-size, distinctly arrow-shaped blobs are accepted from this source; it
# is a recall fallback, not a second opinion on every bright spot.
WHITE_OPEN_UNITS = 0.3
WHITE_MIN_SIDE_FRACTION = 0.65
WHITE_MIN_SOLIDITY = 0.75
WHITE_MAX_SOLIDITY = 0.9
# A soft bright patch on the floor also passes the local threshold, but its rim
# only partly follows a printed outline. A real white arrow has a dark rim, so
# most of its border lies on the edge map.
WHITE_MIN_EDGE_SUPPORT = 0.8
# White paint is brighter than the floor it lies on. A cast shadow of an arrow
# has the same silhouette in the edge map but no brightness step, so it is not
# accepted as paint.
WHITE_MIN_CONTRAST = 20.0
WHITE_RING_UNITS = 0.35

# Chain graph: arrows of one route are close and point roughly along the path.
LINK_MAX_STEP_UNITS = 2.6
LINK_MAX_TURN_DEG = 65.0
LINK_NEIGHBOURS = 2
PRUNE_DETOUR_RATIO = 1.12
# A chain ends at a marker: it never crosses the empty quiet zone around one,
# and each end attaches only to the nearest marker, within this reach.
MARKER_ZONE_UNITS = 1.5
MARKER_ZONE_END_MARGIN = 0.1
MARKER_REACH_UNITS = 3.0
# An interior arrow closer to the frame border than this means the chain runs
# off the picture; only the arrows at a marker may sit that close to the edge.
FRAME_EDGE_UNITS = 0.9

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
    return _routes_from_internal(
        arrows,
        marker_list,
        unit,
        (bgr.shape[0], bgr.shape[1]),
        px_per_meter,
        min_arrows,
    )


def detect_arrows(
    image: object,
    markers: list[Marker] | None = None,
    *,
    dictionary: str = DEFAULT_DICTIONARY,
) -> list[Arrow]:
    """Return arrow candidates with the direction their head points to.

    This is the single-frame half of :func:`detect_arrow_routes`. It is used by
    :mod:`mars.fusion`, which has to vote on individual arrows from several
    photos instead of trusting the route count of one frame.
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
    return [_public_arrow(arrow) for arrow in _find_arrows(bgr, marker_list, unit)]


def routes_from_arrows(
    arrows: Sequence[Arrow],
    markers: list[Marker] | None,
    *,
    px_per_meter: float | None = None,
    frame_shape: tuple[int, int] | None = None,
    min_arrows: int = 2,
) -> list[ArrowRoute]:
    """Build marker-to-marker routes from already detected arrows.

    Frames of the same field are shifted relative to each other, so
    :mod:`mars.fusion` maps their arrows into one reference frame and rebuilds
    the chains here. ``frame_shape`` is the reference frame size ``(height,
    width)``; without it the frame-border rule is disabled.
    """
    if markers is None:
        return []
    marker_list = sorted(markers, key=lambda marker: marker.id)
    if len(marker_list) < 2:
        return []
    unit = float(np.median([marker.side_px for marker in marker_list]))
    if not math.isfinite(unit) or unit <= 0:
        return []
    internal: list[_Arrow] = []
    for index, arrow in enumerate(arrows):
        direction = np.asarray(arrow.direction, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        center = np.asarray(arrow.center, dtype=np.float64)
        if (
            not math.isfinite(norm)
            or norm <= 1e-9
            or center.shape != (2,)
            or not bool(np.all(np.isfinite(center)))
        ):
            continue
        internal.append(
            _Arrow(
                index=index,
                center=center,
                axis=direction / norm,
                head_sign=1,
                area_px=arrow.area_px,
                length_px=arrow.length_px,
                color=arrow.color,
            )
        )
    shape = frame_shape if frame_shape is not None else (10**9, 10**9)
    return _routes_from_internal(
        internal, marker_list, unit, shape, px_per_meter, min_arrows
    )


def _routes_from_internal(
    arrows: list[_Arrow],
    marker_list: list[Marker],
    unit: float,
    frame_shape: tuple[int, int],
    px_per_meter: float | None,
    min_arrows: int,
) -> list[ArrowRoute]:
    """Link arrow candidates into routes; shared by both public detectors."""
    if len(arrows) < min_arrows:
        return []

    adjacency = _link_graph(arrows, marker_list, unit)
    adjacency = _prune_shortcuts(adjacency, arrows)

    chains: list[_Chain] = []
    for component in _components(adjacency):
        if len(component) < min_arrows:
            continue
        path = _order_chain(adjacency, component, arrows)
        if len(path) < min_arrows:
            continue
        chain = _match_markers(path, arrows, marker_list, unit, frame_shape)
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


def _public_arrow(arrow: _Arrow) -> Arrow:
    """Convert an internal candidate so its direction points at the head."""
    axis = arrow.axis if arrow.head_sign >= 0 else -arrow.axis
    norm = float(np.linalg.norm(axis))
    if not math.isfinite(norm) or norm <= 1e-9:
        axis = np.asarray((1.0, 0.0))
    else:
        axis = axis / norm
    return Arrow(
        center=(float(arrow.center[0]), float(arrow.center[1])),
        direction=(float(axis[0]), float(axis[1])),
        area_px=arrow.area_px,
        length_px=arrow.length_px,
        color=arrow.color,
    )


def second_longest_route(routes: list[ArrowRoute]) -> ArrowRoute | None:
    """Return the second longest route, or ``None`` when fewer than two exist.

    Task 2 expects the route that is second by length, so routes are ranked
    here once and the caller does not repeat the ordering.
    """
    ordered = sorted(routes, key=lambda route: route.length_px, reverse=True)
    return ordered[1] if len(ordered) > 1 else None


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
    # closed silhouette whatever colour the arrow itself is. Canny limits follow
    # the frame contrast: a fixed pair loses white arrows when the light floor
    # and the white paint merge, and floods the map with texture in harsh light.
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    canny_high = float(
        np.clip(
            CANNY_SCALE * np.percentile(gradient, CANNY_PERCENTILE),
            CANNY_HIGH_MIN,
            CANNY_HIGH_MAX,
        )
    )
    edges = cv2.Canny(gray, CANNY_LOW_RATIO * canny_high, canny_high)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    # Tolerant edge map for the white-paint check below: the rim of a paint blob
    # is a few pixels wide, so the contour may sit next to an edge, not on it.
    edge_support = cv2.dilate(edges, np.ones((3, 3), np.uint8))

    # Saturated arrows (red/blue) also give solid colour blobs; adding them
    # recovers arrows whose outline is broken by glare. Saturation and value
    # limits are taken from the frame itself so exposure shifts do not hide
    # coloured paint.
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    saturation_high = float(np.percentile(saturation, SATURATION_PERCENTILE))
    value_high = float(np.percentile(value, VALUE_PERCENTILE))
    red_saturation = float(
        np.clip(
            RED_SATURATION_SCALE * saturation_high,
            RED_SATURATION_MIN,
            RED_SATURATION_MAX,
        )
    )
    blue_saturation = float(
        np.clip(
            BLUE_SATURATION_SCALE * saturation_high,
            BLUE_SATURATION_MIN,
            BLUE_SATURATION_MAX,
        )
    )
    value_min = float(
        np.clip(VALUE_MIN_SCALE * value_high, VALUE_MIN_FLOOR, VALUE_MIN_CAP)
    )
    red_mask = (
        ((hue <= 14) | (hue >= 163))
        & (saturation > red_saturation)
        & (value > value_min)
    )
    blue_mask = (
        (hue >= 95)
        & (hue <= 140)
        & (saturation > blue_saturation)
        & (value > value_min)
    )
    sources = [edges]
    for mask in (red_mask, blue_mask):
        filled = (mask.astype(np.uint8) * 255)
        sources.append(cv2.morphologyEx(filled, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)))

    # White arrows need their own source: on a beige or grey floor the edge map
    # may leave them open, while their paint is still the brightest low-
    # saturation area. The adaptive threshold follows the local floor, and the
    # vertical opening removes thin bright seams that would swallow an arrow.
    block = int(round(WHITE_ADAPTIVE_BLOCK_UNITS * unit)) | 1
    block = max(15, min(block, 151))
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        -WHITE_ADAPTIVE_C,
    )
    white = np.where(saturation < WHITE_SATURATION_MAX, adaptive, 0).astype(np.uint8)
    open_height = max(3, int(round(WHITE_OPEN_UNITS * unit)) | 1)
    white = cv2.morphologyEx(
        white,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, open_height)),
    )
    sources.append(white)

    min_area = max(MIN_AREA_FRACTION * unit * unit, MIN_AREA_PX)
    max_area = MAX_AREA_FRACTION * unit * unit
    min_side = MIN_SIDE_FRACTION * unit
    max_side = MAX_SIDE_FRACTION * unit
    margin_x = 0.01 * width
    margin_y = 0.01 * height

    candidates: list[tuple[np.ndarray, int, str, np.ndarray]] = []
    for source_index, source in enumerate(sources):
        # The last source is the white-paint fallback; it accepts only blobs
        # that are both full-size and clearly arrow-shaped.
        white_only = source_index == len(sources) - 1
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
            solidity = area / max(cv2.contourArea(hull), 1.0)
            if solidity < MIN_SOLIDITY:
                continue
            if white_only and not (
                max(w, h) >= WHITE_MIN_SIDE_FRACTION * unit
                and WHITE_MIN_SOLIDITY <= solidity <= WHITE_MAX_SOLIDITY
            ):
                continue
            if white_only and _edge_support(contour, edge_support, gray.shape) < (
                WHITE_MIN_EDGE_SUPPORT
            ):
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
            color = _color_of(bgr, mask)
            if color == COLOUR_WHITE and _local_contrast(gray, contour, unit) < (
                WHITE_MIN_CONTRAST
            ):
                continue
            candidates.append((center, int(area), color, contour))

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


def _local_contrast(gray: np.ndarray, contour: np.ndarray, unit: float) -> float:
    """Brightness step between a contour's fill and the floor around it.

    Printed white paint is brighter than the floor, while the shadow of an
    arrow has an arrow silhouette in the edge map but almost no step. The
    reference ring is the bounding box grown by :data:`WHITE_RING_UNITS`, so it
    follows the local lighting instead of a frame-wide level.
    """
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    x, y, w, h = cv2.boundingRect(contour)
    pad = max(2, int(round(WHITE_RING_UNITS * unit)))
    ring = np.zeros_like(mask)
    ring[max(0, y - pad) : y + h + pad, max(0, x - pad) : x + w + pad] = 255
    ring[mask > 0] = 0
    if not np.any(ring):
        return 0.0
    return float(cv2.mean(gray, mask=mask)[0]) - float(cv2.mean(gray, mask=ring)[0])


def _edge_support(contour: np.ndarray, edges: np.ndarray, shape: tuple[int, ...]) -> float:
    """Fraction of a contour's border that lies on the printed outline.

    A real arrow has a dark rim, so its paint blob is bounded by edges. A soft
    brightness patch on the floor passes the local white threshold too, but
    only part of its rim coincides with an outline; this ratio separates them.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    border = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    pixels = border > 0
    total = int(np.count_nonzero(pixels))
    if total == 0:
        return 0.0
    return float(np.count_nonzero(edges[pixels])) / total


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
    """Classify the arrow colour by channel ratios inside ``mask``.

    Ratios rather than absolute levels keep the label stable when the frame is
    darker or brighter: a shadowed red arrow is still red.
    """
    blue, green, red = (float(channel) for channel in cv2.mean(bgr, mask=mask)[:3])
    if red > 1.35 * max(green, blue) and red > 40:
        return COLOUR_RED
    if blue > 1.15 * max(red, green) and blue > 35:
        return COLOUR_BLUE
    return COLOUR_WHITE


def _link_graph(
    arrows: list[_Arrow], markers: list[Marker], unit: float
) -> dict[int, set[int]]:
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
            if _crosses_marker_zone(first.center, second.center, markers, unit):
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


def _crosses_marker_zone(
    first: np.ndarray, second: np.ndarray, markers: list[Marker], unit: float
) -> bool:
    """Whether a link passes through the empty quiet zone of a marker.

    Consecutive arrows of one route are close, but the ring around a marker is
    unpainted. When two routes terminate at the same marker their ends are
    close as well, so a link across that empty ring would join them. Only the
    middle of a link is checked: an end arrow is expected to sit near a marker.
    """
    delta = second - first
    length_sq = float(delta @ delta)
    if length_sq <= 0.0:
        return True
    for marker in markers:
        center = np.asarray(marker.center, dtype=np.float64)
        t = float((center - first) @ delta) / length_sq
        if not MARKER_ZONE_END_MARGIN <= t <= 1.0 - MARKER_ZONE_END_MARGIN:
            continue
        closest = first + t * delta
        if float(np.linalg.norm(center - closest)) < MARKER_ZONE_UNITS * unit:
            return True
    return False


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
    path: list[int],
    arrows: list[_Arrow],
    markers: list[Marker],
    unit: float,
    frame_shape: tuple[int, int],
) -> _Chain | None:
    """Attach both chain ends to the markers nearest to them.

    Only the nearest marker can own an end. A generous "any marker within
    reach" rule attaches a chain cut by the frame edge to whatever marker
    happens to be on that side, which invents a route between two markers that
    the chain never connected.

    A chain whose interior arrows run into the frame border is also rejected:
    a printed route leaves the picture only at a marker, so an arrow that close
    to the edge is a fragment (a cut marker ring, a clipped arrow) and the chain
    that contains it has most likely lost its rest outside the frame.
    """
    by_index = {arrow.index: arrow for arrow in arrows}
    head = by_index[path[0]]
    tail = by_index[path[-1]]
    start, head_gap = _nearest_marker(head.center, markers)
    end, tail_gap = _nearest_marker(tail.center, markers)
    if start is None or end is None or start.id == end.id:
        return None
    if head_gap > MARKER_REACH_UNITS * unit or tail_gap > MARKER_REACH_UNITS * unit:
        return None
    for index in path[1:-1]:
        if _border_gap(by_index[index].center, frame_shape) < FRAME_EDGE_UNITS * unit:
            return None
    return _Chain(start=start, end=end, arrows=[by_index[index] for index in path])


def _border_gap(center: np.ndarray, frame_shape: tuple[int, int]) -> float:
    """Shortest distance from ``center`` to the frame border, in pixels."""
    height, width = frame_shape
    x, y = float(center[0]), float(center[1])
    return min(x, y, width - x, height - y)


def _nearest_marker(
    center: np.ndarray, markers: list[Marker]
) -> tuple[Marker | None, float]:
    """Return the marker closest to ``center`` and the distance to it."""
    best: Marker | None = None
    best_gap = math.inf
    for marker in markers:
        gap = float(np.linalg.norm(center - np.asarray(marker.center, dtype=np.float64)))
        if gap < best_gap:
            best, best_gap = marker, gap
    return best, best_gap


def _to_public(chain: _Chain, px_per_meter: float | None, unit: float) -> ArrowRoute:
    """Convert an internal chain into a public route with a metric length."""
    arrows = list(chain.arrows)
    start_center = np.asarray(chain.start.center, dtype=np.float64)
    end_center = np.asarray(chain.end.center, dtype=np.float64)

    # Majority vote of arrowheads decides the travel direction. Travel runs
    # towards the next arrow, and past the last arrow away from the previous
    # one, so every vote describes the same route direction.
    votes = 0
    for index, arrow in enumerate(arrows):
        if index + 1 < len(arrows):
            travel = arrows[index + 1].center - arrow.center
        else:
            travel = arrow.center - arrows[index - 1].center
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
