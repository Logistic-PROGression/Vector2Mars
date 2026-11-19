"""Offline tests for ArUco detection and pixel-to-metre scale estimation."""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

from mars.aruco import DEFAULT_DICTIONARY, Marker, annotate_markers, detect_markers
from mars.scale import px_per_meter

REPO_ROOT = Path(__file__).resolve().parents[1]
SATELLITE = REPO_ROOT / "samples" / "image.jpg"


def _marker_image(
    marker_id: int = 7,
    size_px: int = 120,
    margin: int = 40,
    dictionary: str = DEFAULT_DICTIONARY,
) -> np.ndarray:
    """White frame with one generated marker, quiet zone included."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_px)
    frame = np.full(
        (size_px + 2 * margin, size_px + 2 * margin), 255, dtype=np.uint8
    )
    frame[margin:margin + size_px, margin:margin + size_px] = marker
    return frame


def _two_markers_image() -> np.ndarray:
    """White canvas with markers 7 and 3 in separate quiet zones."""
    frame = np.full((360, 360), 255, dtype=np.uint8)
    aruco_dict = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, DEFAULT_DICTIONARY)
    )
    for marker_id, (x, y) in ((7, (20, 20)), (3, (200, 200))):
        marker = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 100)
        frame[y:y + 100, x:x + 100] = marker
    return frame


class DetectMarkersTest(unittest.TestCase):
    def test_detects_id_center_and_side(self):
        markers = detect_markers(_marker_image())

        self.assertEqual([marker.id for marker in markers], [7])
        marker = markers[0]
        self.assertAlmostEqual(marker.center[0], 100.0, delta=5.0)
        self.assertAlmostEqual(marker.center[1], 100.0, delta=5.0)
        self.assertAlmostEqual(marker.side_px, 120.0, delta=12.0)
        self.assertEqual(marker.corners.shape, (4, 2))

    def test_multiple_markers_are_sorted_by_id(self):
        markers = detect_markers(_two_markers_image())
        self.assertEqual([marker.id for marker in markers], [3, 7])

    def test_bgr_and_grayscale_are_accepted(self):
        gray = _marker_image()
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self.assertEqual([m.id for m in detect_markers(bgr)], [7])

    def test_unsupported_inputs_return_empty_list(self):
        for image in (
            None,
            np.zeros((0, 0), dtype=np.uint8),
            np.full((100, 100), 255, dtype=np.float32),
            np.zeros((10, 10, 5), dtype=np.uint8),
        ):
            with self.subTest(image_type=type(image).__name__):
                self.assertEqual(detect_markers(image), [])

    def test_blank_frame_returns_empty_list(self):
        self.assertEqual(
            detect_markers(np.full((120, 160), 255, dtype=np.uint8)), []
        )

    def test_unknown_dictionary_raises_value_error(self):
        with self.assertRaises(ValueError):
            detect_markers(_marker_image(), "DICT_NOT_REAL")

    def test_input_is_not_mutated(self):
        frame = _marker_image()
        snapshot = frame.copy()
        detect_markers(frame)
        np.testing.assert_array_equal(frame, snapshot)

    def test_detected_marker_reports_positive_side(self):
        marker = detect_markers(_marker_image())[0]
        self.assertGreater(marker.side_px, 0.0)


class AnnotateMarkersTest(unittest.TestCase):
    def test_overlay_keeps_shape_and_changes_pixels(self):
        frame = _marker_image()
        overlay = annotate_markers(frame, detect_markers(frame))
        # Grayscale input is promoted to BGR, matching ``mars.vision.annotate``.
        self.assertEqual(overlay.shape[:2], frame.shape[:2])
        self.assertEqual(overlay.dtype, np.uint8)
        self.assertFalse(np.array_equal(overlay, frame))

    def test_none_image_and_markers_are_tolerated(self):
        overlay = annotate_markers(None, None)
        self.assertEqual(overlay.ndim, 3)
        self.assertGreater(overlay.size, 0)

    def test_input_is_not_mutated(self):
        frame = _marker_image()
        snapshot = frame.copy()
        annotate_markers(frame, detect_markers(frame))
        np.testing.assert_array_equal(frame, snapshot)


class ScaleTest(unittest.TestCase):
    @staticmethod
    def _marker(side_px: float) -> Marker:
        return Marker(
            id=0,
            corners=np.zeros((4, 2), dtype=np.float32),
            center=(0.0, 0.0),
            side_px=side_px,
        )

    def test_scale_from_single_marker(self):
        self.assertAlmostEqual(px_per_meter([self._marker(100.0)], 0.1), 1000.0)

    def test_median_ignores_one_bad_corner(self):
        markers = [self._marker(90.0), self._marker(100.0), self._marker(110.0)]
        self.assertAlmostEqual(px_per_meter(markers, 0.1), 1000.0)

    def test_no_markers_returns_none(self):
        self.assertIsNone(px_per_meter([], 0.1))
        self.assertIsNone(px_per_meter(None, 0.1))

    def test_non_positive_marker_size_raises(self):
        for size in (0.0, -0.1):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    px_per_meter([self._marker(100.0)], size)


@unittest.skipUnless(SATELLITE.exists(), "satellite sample is not present")
class SatelliteImageTest(unittest.TestCase):
    def test_finds_both_markers_on_satellite_image(self):
        image = cv2.imread(str(SATELLITE))
        self.assertIsNotNone(image)
        markers = detect_markers(image)
        self.assertEqual([marker.id for marker in markers], [0, 2])
        for marker in markers:
            self.assertGreater(marker.side_px, 0.0)


if __name__ == "__main__":
    unittest.main()
