"""Vote on individual arrows across several photos of the same field.

The stop photos of a photo mission show the same field from slightly different
positions. One frame can lose a faded arrow, clip a route at the border, or
mistake a bright patch for paint. Counting routes in every frame and voting on
the totals does not fix that: misses are one-sided, so three frames that each
lose a different arrow still vote for the wrong count.

Instead the frames are combined arrow by arrow:

1. the sharpest frame in the largest marker set shared by at least two pictures
   becomes the reference;
2. every other frame is mapped onto it by a homography fitted to the ArUco
   marker centres and checked against the marker corners, so a wrong detection
   or a tilted frame cannot shift the arrows;
3. arrows of all frames are clustered in reference coordinates; a cluster is a
   real arrow only when enough frames support it (2 of 3, 3 of 5, ...);
4. chains are rebuilt on the confirmed arrows by :mod:`mars.arrow_routes`.

A single frame, a failed homography, or too few confirmed arrows falls back to
the previous behaviour: report the routes of the best individual frame.
"""

# A hint with the expected arrow counts per route is optional and stays outside
# the vote: :func:`select_by_arrow_hint` keeps only frames whose counts match
# and returns the input unchanged when nothing matches.

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from mars.arrow_routes import Arrow, ArrowRoute, routes_from_arrows
from mars.aruco import Marker

# Arrows that land this close in the reference frame are the same arrow.
# Within one frame a split silhouette can be detected twice; the direction
# check keeps two genuinely different arrows from being merged.
DEDUPE_UNITS = 0.4
DEDUPE_MIN_DOT = 0.7
CLUSTER_UNITS = 0.45

# Homography validation. The field is planar and the camera looks down, so the
# marker centres fix the mapping exactly; if one centre is wrong, the marker
# corners stop matching and the frame is skipped.
HOMOGRAPHY_MIN_SCALE = 0.4
HOMOGRAPHY_MAX_SCALE = 2.5
HOMOGRAPHY_CORNER_UNITS = 0.35
MIN_HOMOGRAPHY_MARKERS = 4

MIN_FRAMES_FOR_FUSION = 2

# The field map is pinned by four ArUco markers; a frame with fewer of them is
# clipped and cannot be compared with the expected route counts.
ARROW_HINT_MARKERS = 4


@dataclass(frozen=True)
class FrameObservation:
    """One analysed photo: markers, arrows and its own route candidates."""

    path: Path | None
    image: np.ndarray = field(repr=False, compare=False)
    markers: tuple[Marker, ...] = ()
    arrows: tuple[Arrow, ...] = ()
    routes: tuple[ArrowRoute, ...] = ()
    sharpness: float = 0.0

    @property
    def marker_ids(self) -> frozenset[int]:
        return frozenset(marker.id for marker in self.markers)

    @property
    def arrow_count(self) -> int:
        """Arrows that the single-frame detector put into routes."""
        return sum(len(route.arrows) for route in self.routes)


@dataclass(frozen=True)
class FusionResult:
    """Routes chosen from one frame or from the consensus of several."""

    routes: tuple[ArrowRoute, ...] = ()
    reference: FrameObservation | None = None
    used: tuple[FrameObservation, ...] = ()
    frame_counts: tuple[int, ...] = ()
    confirmed: int = 0
    dropped: int = 0
    mode: str = "none"  # "fusion", "single" or "none"
    note: str = ""


def fuse_routes(
    observations: list[FrameObservation], *, px_per_meter: float | None = None
) -> FusionResult:
    """Pick the route set supported by most frames.

    ``observations`` are analysed stop photos in flight order. Frames without
    markers are ignored. The result always points at a reference frame, so the
    caller can annotate it without knowing which mode was used.
    """
    usable = [obs for obs in observations if obs.image is not None and obs.markers]
    if not usable:
        return FusionResult(mode="none", note="нет кадров с метками")

    group = _choose_group(usable)
    if group is None:
        return _best_single(usable, "нет общих меток на двух кадрах")
    marker_ids, members = group
    reference = max(members, key=lambda obs: (obs.sharpness, obs.arrow_count))
    reference_markers = sorted(reference.markers, key=lambda marker: marker.id)
    unit = float(np.median([marker.side_px for marker in reference_markers]))
    if not math.isfinite(unit) or unit <= 0:
        return _best_single(usable, "не удалось определить размер метки")

    mapped: list[tuple[int, Arrow]] = []
    used: list[FrameObservation] = []
    for observation in members:
        if observation is reference:
            matrix = np.eye(3, dtype=np.float64)
        else:
            matrix = _fit_homography(observation.markers, reference_markers, unit)
            if matrix is None:
                continue
        frame_arrows = _dedupe(observation.arrows, DEDUPE_UNITS * unit)
        if observation is reference:
            converted = frame_arrows
        else:
            converted = [
                mapped_arrow
                for arrow in frame_arrows
                if (mapped_arrow := _map_arrow(arrow, matrix)) is not None
            ]
        frame_index = len(used)
        mapped.extend((frame_index, arrow) for arrow in converted)
        used.append(observation)

    if len(used) < MIN_FRAMES_FOR_FUSION:
        return _best_single(usable, "гомография не сошлась на соседних кадрах")

    clusters = _cluster(mapped, CLUSTER_UNITS * unit)
    min_support = _min_support(len(used))
    confirmed_arrows: list[Arrow] = []
    dropped = 0
    for cluster in clusters:
        by_frame: dict[int, Arrow] = {}
        for frame_index, arrow in cluster:
            previous = by_frame.get(frame_index)
            if previous is None or arrow.area_px > previous.area_px:
                by_frame[frame_index] = arrow
        if len(by_frame) < min_support:
            dropped += 1
            continue
        confirmed_arrows.append(_fuse_cluster(by_frame))

    routes = tuple(
        routes_from_arrows(
            confirmed_arrows,
            reference_markers,
            px_per_meter=px_per_meter,
            frame_shape=(reference.image.shape[0], reference.image.shape[1]),
        )
    )
    if len(confirmed_arrows) < 2 or not routes:
        return _best_single(
            usable, f"подтверждено стрелок: {len(confirmed_arrows)}"
        )

    confirmed = sum(len(route.arrows) for route in routes)
    return FusionResult(
        routes=routes,
        reference=reference,
        used=tuple(used),
        frame_counts=tuple(obs.arrow_count for obs in used),
        confirmed=confirmed,
        dropped=dropped,
        mode="fusion",
        note=f"общих меток: {sorted(marker_ids)}",
    )


def select_by_arrow_hint(
    observations: list[FrameObservation], expected: Sequence[int]
) -> list[FrameObservation]:
    """Prefer frames whose route arrow counts match the expected set.

    Only frames with the full marker set take part: with fewer markers the
    field is clipped and the route count cannot be compared. ``expected`` is
    order-independent. When no frame matches, the original list is returned
    unchanged so the caller can report the usual result.
    """
    wanted = sorted(expected)
    matching = [
        observation
        for observation in observations
        if len(observation.markers) == ARROW_HINT_MARKERS
        and sorted(len(route.arrows) for route in observation.routes) == wanted
    ]
    return matching or observations


def _choose_group(
    observations: list[FrameObservation],
) -> tuple[frozenset[int], list[FrameObservation]] | None:
    """Find the largest marker set shared by the most frames.

    The reference frame should keep as many markers as possible, but a single
    frame with an extra marker must not win over a marker set that several
    frames share: the fusion needs at least two pictures.
    """
    id_sets = [obs.marker_ids for obs in observations]
    candidates: set[frozenset[int]] = set()
    for ids in id_sets:
        if len(ids) >= MIN_HOMOGRAPHY_MARKERS:
            candidates.add(ids)
    for first in range(len(id_sets)):
        for second in range(first + 1, len(id_sets)):
            common = id_sets[first] & id_sets[second]
            if len(common) >= MIN_HOMOGRAPHY_MARKERS:
                candidates.add(common)

    best: tuple[tuple[int, int, float], frozenset[int], list[FrameObservation]] | None = None
    for ids in candidates:
        members = [obs for obs in observations if ids <= obs.marker_ids]
        if len(members) < MIN_FRAMES_FOR_FUSION:
            continue
        score = (len(members), len(ids), max(obs.sharpness for obs in members))
        if best is None or score > best[0]:
            best = (score, ids, members)
    if best is None:
        return None
    return best[1], best[2]


def _fit_homography(
    source_markers: tuple[Marker, ...] | list[Marker],
    target_markers: list[Marker],
    unit: float,
) -> np.ndarray | None:
    """Map one frame onto the reference frame through shared markers.

    Four centres are enough to fit the homography exactly, which also means a
    wrong centre is not visible in the fit residual. The marker corners give
    the independent check: both frames see the same printed squares, so after
    a correct mapping the predicted corners must land on the detected ones.
    """
    target_by_id = {marker.id: marker for marker in target_markers}
    pairs = [
        (marker, target_by_id[marker.id])
        for marker in source_markers
        if marker.id in target_by_id
    ]
    if len(pairs) < MIN_HOMOGRAPHY_MARKERS:
        return None

    source = np.asarray([marker.center for marker, _ in pairs], dtype=np.float32)
    target = np.asarray([marker.center for _, marker in pairs], dtype=np.float32)
    try:
        matrix, _ = cv2.findHomography(source, target, 0)
    except cv2.error:
        return None
    if matrix is None or not bool(np.all(np.isfinite(matrix))):
        return None
    if abs(float(np.linalg.det(matrix))) < 1e-12:
        return None

    affine_det = float(
        matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    )
    if affine_det <= 0:
        return None
    scale = math.sqrt(affine_det)
    if not HOMOGRAPHY_MIN_SCALE <= scale <= HOMOGRAPHY_MAX_SCALE:
        return None

    errors: list[float] = []
    for source_marker, target_marker in pairs:
        predicted = _map_points(source_marker.corners, matrix)
        errors.append(
            float(
                np.mean(
                    np.linalg.norm(predicted - target_marker.corners, axis=1)
                )
            )
        )
    if errors and max(errors) > HOMOGRAPHY_CORNER_UNITS * unit:
        return None
    return matrix


def _map_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32).reshape(1, -1, 2)
    return cv2.perspectiveTransform(array, matrix)[0].astype(np.float64)


def _map_arrow(arrow: Arrow, matrix: np.ndarray) -> Arrow | None:
    """Move one arrow into reference coordinates, direction included."""
    center = np.asarray(arrow.center, dtype=np.float64)
    direction = np.asarray(arrow.direction, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm <= 1e-9:
        return None
    direction = direction / norm

    mapped_center = _map_points(center.reshape(1, 2), matrix)[0]
    mapped_tip = _map_points((center + direction).reshape(1, 2), matrix)[0]
    vector = mapped_tip - mapped_center
    length = float(np.linalg.norm(vector))
    if not math.isfinite(length) or length <= 1e-9:
        return None
    vector = vector / length
    return Arrow(
        center=(float(mapped_center[0]), float(mapped_center[1])),
        direction=(float(vector[0]), float(vector[1])),
        area_px=arrow.area_px,
        length_px=arrow.length_px * length,
        color=arrow.color,
    )


def _dedupe(arrows: tuple[Arrow, ...], radius: float) -> list[Arrow]:
    """Drop a second detection of the same physical arrow in one frame."""
    kept: list[Arrow] = []
    for arrow in sorted(arrows, key=lambda item: item.area_px, reverse=True):
        center = np.asarray(arrow.center, dtype=np.float64)
        direction = np.asarray(arrow.direction, dtype=np.float64)
        duplicate = False
        for other in kept:
            gap = float(
                np.linalg.norm(
                    center - np.asarray(other.center, dtype=np.float64)
                )
            )
            if gap > radius:
                continue
            dot = float(
                direction @ np.asarray(other.direction, dtype=np.float64)
            )
            if abs(dot) >= DEDUPE_MIN_DOT:
                duplicate = True
                break
        if not duplicate:
            kept.append(arrow)
    return kept


def _cluster(
    mapped: list[tuple[int, Arrow]], radius: float
) -> list[list[tuple[int, Arrow]]]:
    """Group arrows of different frames that land on the same spot."""
    clusters: list[list[tuple[int, Arrow]]] = []
    for frame_index, arrow in mapped:
        center = np.asarray(arrow.center, dtype=np.float64)
        placed = False
        for cluster in clusters:
            # One physical arrow is printed once, so a cluster holds at most
            # one detection per frame. A second candidate from the same frame
            # is a split silhouette, not the same vote.
            if any(item_frame == frame_index for item_frame, _ in cluster):
                continue
            centroid = np.mean(
                [
                    np.asarray(item.center, dtype=np.float64)
                    for _, item in cluster
                ],
                axis=0,
            )
            if float(np.linalg.norm(center - centroid)) > radius:
                continue
            cluster.append((frame_index, arrow))
            placed = True
            break
        if not placed:
            clusters.append([(frame_index, arrow)])
    return clusters


def _fuse_cluster(members: dict[int, Arrow]) -> Arrow:
    """Average one confirmed arrow over the frames that saw it."""
    arrows = list(members.values())
    center = np.mean(
        [np.asarray(arrow.center, dtype=np.float64) for arrow in arrows], axis=0
    )
    # The head detector is weak on short or blurred arrows and sometimes
    # reports the opposite direction. Align every member with the largest
    # arrow before averaging; the route still decides its direction by a
    # majority of all arrowheads.
    largest = max(arrows, key=lambda arrow: arrow.area_px)
    reference = np.asarray(largest.direction, dtype=np.float64)
    vectors = [
        np.asarray(arrow.direction, dtype=np.float64)
        for arrow in arrows
    ]
    vectors = [
        vector if float(vector @ reference) >= 0 else -vector
        for vector in vectors
    ]
    direction = np.mean(vectors, axis=0)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-9:
        direction = reference
    else:
        direction = direction / norm

    colors = Counter(arrow.color for arrow in arrows)
    color = min(colors, key=lambda item: (-colors[item], item))
    areas = sorted(arrow.area_px for arrow in arrows)
    lengths = sorted(arrow.length_px for arrow in arrows)
    middle = len(arrows) // 2
    return Arrow(
        center=(float(center[0]), float(center[1])),
        direction=(float(direction[0]), float(direction[1])),
        area_px=int(areas[middle]),
        length_px=float(lengths[middle]),
        color=color,
    )


def _min_support(frames: int) -> int:
    """How many frames must see an arrow: 2 of 3, 3 of 5, 3 of 4."""
    return max(2, frames // 2 + 1)


def _best_single(observations: list[FrameObservation], note: str) -> FusionResult:
    """Fallback: keep the routes of the strongest individual frame."""
    best = max(
        observations,
        key=lambda obs: (len(obs.routes), obs.arrow_count, obs.sharpness),
    )
    routes = tuple(best.routes)
    confirmed = sum(len(route.arrows) for route in routes)
    return FusionResult(
        routes=routes,
        reference=best,
        used=(best,),
        frame_counts=(confirmed,),
        confirmed=confirmed,
        dropped=0,
        mode="single",
        note=note,
    )
