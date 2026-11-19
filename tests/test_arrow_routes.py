"""Offline tests for arrow-route detection between ArUco markers."""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

from mars.aruco import detect_markers
from mars.arrow_routes import annotate_arrow_routes, detect_arrow_routes

REPO_ROOT = Path(__file__).resolve().parents[1]
SATELLITE = REPO_ROOT / "samples" / "image.jpg"

MARKER_SIZE = 80
MARKER_DICTIONARY = "DICT_4X4_50"
BACKGROUND = 70


def _canvas(width: int = 900, height: int = 700) -> np.ndarray:
    return np.full((height, width, 3), BACKGROUND, dtype=np.uint8)


def _paste_marker(
    canvas: np.ndarray, marker_id: int, center: tuple[int, int]
) -> None:
    """Draw an ArUco marker with a white quiet zone."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, MARKER_DICTIONARY)
    )
    marker = cv2.cvtColor(
        cv2.aruco.generateImageMarker(aruco_dict, marker_id, MARKER_SIZE),
        cv2.COLOR_GRAY2BGR,
    )
    quiet = MARKER_SIZE + 20
    x, y = center
    cv2.rectangle(
        canvas,
        (x - quiet // 2, y - quiet // 2),
        (x + quiet // 2, y + quiet // 2),
        (255, 255, 255),
        -1,
    )
    half = MARKER_SIZE // 2
    canvas[y - half:y + half, x - half:x + half] = marker


def _draw_arrows(
    canvas: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    count: int = 3,
) -> None:
    """Draw white arrows along the segment, pointing from start to end."""
    first = np.asarray(start, dtype=np.float64)
    last = np.asarray(end, dtype=np.float64)
    direction = (last - first) / np.linalg.norm(last - first)
    for fraction in np.linspace(0.3, 0.7, count):
        base = first + (last - first) * fraction
        tip = base + direction * 40
        cv2.arrowedLine(
            canvas,
            tuple(base.astype(int)),
            tuple(tip.astype(int)),
            (255, 255, 255),
            6,
            tipLength=0.5,
        )


def _single_route(reverse: bool = False) -> np.ndarray:
    canvas = _canvas(600, 700)
    _paste_marker(canvas, 1, (120, 600))
    _paste_marker(canvas, 5, (380, 150))
    if reverse:
        _draw_arrows(canvas, (380, 150), (120, 600))
    else:
        _draw_arrows(canvas, (120, 600), (380, 150))
    return canvas


def _two_routes() -> np.ndarray:
    canvas = _canvas()
    _paste_marker(canvas, 1, (120, 600))
    _paste_marker(canvas, 5, (380, 150))
    _paste_marker(canvas, 2, (520, 600))
    _paste_marker(canvas, 6, (780, 150))
    _draw_arrows(canvas, (120, 600), (380, 150))
    _draw_arrows(canvas, (520, 600), (780, 150))
    return canvas


def _draw_arc_arrows(
    canvas: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    sag: float = 90.0,
    count: int = 5,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """Draw arrows along a curved (quadratic) path, tangent to the curve."""
    first = np.asarray(start, dtype=np.float64)
    last = np.asarray(end, dtype=np.float64)
    axis = last - first
    normal = np.asarray((-axis[1], axis[0])) / np.linalg.norm(axis)
    control = (first + last) / 2 + normal * sag * 2
    for t in np.linspace(0.2, 0.8, count):
        point = (1 - t) ** 2 * first + 2 * (1 - t) * t * control + t**2 * last
        tangent = 2 * (1 - t) * (control - first) + 2 * t * (last - control)
        tangent /= np.linalg.norm(tangent)
        tip = point + tangent * 40
        cv2.arrowedLine(
            canvas,
            tuple(point.astype(int)),
            tuple(tip.astype(int)),
            color,
            6,
            tipLength=0.5,
        )


def _mixed_colour_route() -> np.ndarray:
    canvas = _canvas(600, 700)
    _paste_marker(canvas, 1, (120, 600))
    _paste_marker(canvas, 5, (380, 150))
    first = np.asarray((120, 600), dtype=np.float64)
    last = np.asarray((380, 150), dtype=np.float64)
    direction = (last - first) / np.linalg.norm(last - first)
    colours = ((255, 255, 255), (0, 0, 255), (255, 0, 0))
    for fraction, color in zip(np.linspace(0.3, 0.7, 3), colours):
        base = first + (last - first) * fraction
        cv2.arrowedLine(
            canvas,
            tuple(base.astype(int)),
            tuple((base + direction * 40).astype(int)),
            color,
            6,
            tipLength=0.5,
        )
    return canvas


class DetectArrowRoutesTest(unittest.TestCase):
    def test_detects_one_route_between_its_markers(self):
        image = _single_route()
        markers = detect_markers(image)
        routes = detect_arrow_routes(image, markers)

        self.assertEqual(len(routes), 1)
        route = routes[0]
        self.assertEqual((route.start_marker_id, route.end_marker_id), (1, 5))
        self.assertEqual(len(route.arrows), 3)
        self.assertGreater(route.length_px, 0.0)
        self.assertEqual(route.polyline[0], tuple(markers[0].center))
        # All arrows point from the start marker to the end marker.
        direction = np.subtract(route.polyline[-1], route.polyline[0])
        direction = direction / np.linalg.norm(direction)
        for arrow in route.arrows:
            self.assertGreater(float(np.asarray(arrow.direction) @ direction), 0.8)

    def test_two_routes_do_not_merge(self):
        routes = detect_arrow_routes(_two_routes())

        pairs = {
            frozenset((route.start_marker_id, route.end_marker_id))
            for route in routes
        }
        self.assertEqual(pairs, {frozenset((1, 5)), frozenset((2, 6))})
        self.assertEqual([len(route.arrows) for route in routes], [3, 3])

    def test_route_direction_follows_arrowheads(self):
        route = detect_arrow_routes(_single_route(reverse=True))[0]
        self.assertEqual((route.start_marker_id, route.end_marker_id), (5, 1))

    def test_px_per_meter_adds_length_in_metres(self):
        route = detect_arrow_routes(_single_route(), px_per_meter=100.0)[0]
        self.assertIsNotNone(route.length_m)
        self.assertAlmostEqual(route.length_m, route.length_px / 100.0)

    def test_curved_route_is_followed(self):
        canvas = _canvas(600, 700)
        _paste_marker(canvas, 1, (120, 600))
        _paste_marker(canvas, 5, (380, 150))
        _draw_arc_arrows(canvas, (120, 600), (380, 150), sag=90, count=5)

        routes = detect_arrow_routes(canvas)

        self.assertEqual(len(routes), 1)
        route = routes[0]
        self.assertEqual(frozenset((route.start_marker_id, route.end_marker_id)), frozenset((1, 5)))
        self.assertEqual(len(route.arrows), 5)
        straight = float(np.linalg.norm(np.subtract((380, 150), (120, 600))))
        self.assertGreater(route.length_px, straight)

    def test_mixed_colour_arrows_form_one_route(self):
        route = detect_arrow_routes(_mixed_colour_route())[0]
        self.assertEqual(len(route.arrows), 3)
        self.assertEqual(set(route.colors), {"white", "red", "blue"})

    def test_different_arrow_sizes_form_one_route(self):
        canvas = _canvas(600, 700)
        _paste_marker(canvas, 1, (120, 600))
        _paste_marker(canvas, 5, (380, 150))
        first = np.asarray((120, 600), dtype=np.float64)
        last = np.asarray((380, 150), dtype=np.float64)
        direction = (last - first) / np.linalg.norm(last - first)
        for fraction, length in zip(np.linspace(0.3, 0.7, 3), (26, 44, 64)):
            base = first + (last - first) * fraction
            cv2.arrowedLine(
                canvas,
                tuple(base.astype(int)),
                tuple((base + direction * length).astype(int)),
                (255, 255, 255),
                6,
                tipLength=0.5,
            )

        routes = detect_arrow_routes(canvas)

        self.assertEqual(len(routes), 1)
        self.assertEqual(len(routes[0].arrows), 3)

    def test_small_noise_is_ignored(self):
        canvas = _single_route()
        for x, y in ((50, 50), (300, 650), (550, 400), (80, 300)):
            cv2.circle(canvas, (x, y), 2, (255, 255, 255), -1)
        routes = detect_arrow_routes(canvas)
        self.assertEqual(len(routes), 1)
        self.assertEqual(len(routes[0].arrows), 3)

    def test_no_markers_returns_empty_list(self):
        canvas = _canvas(600, 700)
        _draw_arrows(canvas, (100, 600), (500, 100))
        self.assertEqual(detect_arrow_routes(canvas), [])

    def test_unsupported_images_return_empty_list(self):
        for image in (
            None,
            np.zeros((0, 0, 3), dtype=np.uint8),
            np.full((100, 100), 255, dtype=np.float32),
            np.zeros((20, 20, 5), dtype=np.uint8),
        ):
            with self.subTest(image_type=type(image).__name__):
                self.assertEqual(detect_arrow_routes(image), [])

    def test_blank_frame_returns_empty_list(self):
        self.assertEqual(detect_arrow_routes(_canvas()), [])

    def test_input_is_not_mutated(self):
        canvas = _single_route()
        snapshot = canvas.copy()
        detect_arrow_routes(canvas)
        np.testing.assert_array_equal(canvas, snapshot)


class AnnotateArrowRoutesTest(unittest.TestCase):
    def test_overlay_draws_routes(self):
        canvas = _single_route()
        markers = detect_markers(canvas)
        routes = detect_arrow_routes(canvas, markers)
        overlay = annotate_arrow_routes(canvas, routes, markers)
        self.assertEqual(overlay.shape, canvas.shape)
        self.assertEqual(overlay.dtype, np.uint8)
        self.assertFalse(np.array_equal(overlay, canvas))

    def test_none_inputs_are_tolerated(self):
        overlay = annotate_arrow_routes(None, None, None)
        self.assertEqual(overlay.ndim, 3)
        self.assertGreater(overlay.size, 0)

    def test_input_is_not_mutated(self):
        canvas = _single_route()
        snapshot = canvas.copy()
        markers = detect_markers(canvas)
        annotate_arrow_routes(canvas, detect_arrow_routes(canvas, markers), markers)
        np.testing.assert_array_equal(canvas, snapshot)


@unittest.skipUnless(SATELLITE.exists(), "satellite sample is not present")
class SatelliteArrowRouteTest(unittest.TestCase):
    def test_satellite_route_connects_markers_0_and_2(self):
        image = cv2.imread(str(SATELLITE))
        self.assertIsNotNone(image)
        routes = detect_arrow_routes(image)
        self.assertEqual(len(routes), 1)
        route = routes[0]
        self.assertEqual((route.start_marker_id, route.end_marker_id), (0, 2))
        self.assertEqual(len(route.arrows), 3)


if __name__ == "__main__":
    unittest.main()
