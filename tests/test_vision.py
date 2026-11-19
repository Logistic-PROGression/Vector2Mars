"""Offline tests for the route-vision pipeline (stdlib unittest only)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from mars.config import RouteSpec, SceneConfig, load_config
from mars.vision import SceneResult, analyze_frame, annotate, skeleton_length_px

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
BACKGROUND = (40, 40, 40)
THICKNESS = 9  # thicker than the 5x5 morphology kernel so strokes survive

TEST_CONFIG = SceneConfig(
    routes=(
        RouteSpec(
            "red",
            1,
            (((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255))),
        ),
        RouteSpec("green", 2, (((40, 120, 70), (85, 255, 255)),)),
        RouteSpec(
            "blue",
            3,
            (((95, 120, 70), (130, 255, 255)),),
            dangerous=True,
        ),
    ),
    px_per_meter=None,
    min_area_px=50,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _canvas(height: int = 480, width: int = 640) -> np.ndarray:
    return np.full((height, width, 3), BACKGROUND, dtype=np.uint8)


def _line_frame(color_bgr: tuple[int, int, int], length_px: int, y: int = 240):
    """Draw one horizontal stroke of the requested pixel length."""
    frame = _canvas()
    cv2.line(frame, (40, y), (40 + length_px, y), color_bgr, THICKNESS)
    return frame


def _hsv_mask(frame: np.ndarray, lower, upper) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(
        hsv,
        np.asarray(lower, dtype=np.uint8),
        np.asarray(upper, dtype=np.uint8),
    )


class AnalyzeFrameTest(unittest.TestCase):
    def test_shortest_route_excludes_dangerous(self):
        frame = _canvas()
        cv2.line(frame, (40, 100), (240, 100), RED, THICKNESS)  # 200 px
        cv2.line(frame, (40, 240), (140, 240), GREEN, THICKNESS)  # 100 px
        cv2.line(frame, (40, 380), (80, 380), BLUE, THICKNESS)  # 40 px

        result = analyze_frame(frame, TEST_CONFIG)

        self.assertIsNotNone(result.shortest)
        self.assertEqual(result.shortest.name, "green")
        self.assertFalse(result.shortest.dangerous)
        self.assertEqual([route.name for route in result.excluded], ["blue"])
        self.assertEqual(
            sorted(route.name for route in result.routes),
            ["blue", "green", "red"],
        )
        by_name = {route.name: route for route in result.routes}
        self.assertGreater(by_name["red"].length_px, by_name["green"].length_px)
        self.assertGreater(by_name["green"].length_px, by_name["blue"].length_px)
        self.assertIsNone(result.shortest.length_m)

    def test_blank_frame_returns_empty_result(self):
        result = analyze_frame(_canvas(), TEST_CONFIG)
        self.assertEqual(result.routes, [])
        self.assertIsNone(result.shortest)
        self.assertEqual(result.excluded, [])

    def test_none_frame_returns_empty_result(self):
        result = analyze_frame(None, TEST_CONFIG)
        self.assertIsInstance(result, SceneResult)
        self.assertEqual(result.routes, [])
        self.assertIsNone(result.shortest)

    def test_grayscale_frame_is_accepted(self):
        gray = cv2.cvtColor(_line_frame(GREEN, 100), cv2.COLOR_BGR2GRAY)
        result = analyze_frame(gray, TEST_CONFIG)  # must not raise
        self.assertIsInstance(result, SceneResult)
        self.assertIsNone(result.shortest)

    def test_uniform_frame_returns_empty_result(self):
        frame = np.full((240, 320, 3), 90, dtype=np.uint8)
        result = analyze_frame(frame, TEST_CONFIG)
        self.assertEqual(result.routes, [])

    def test_empty_array_returns_empty_result(self):
        result = analyze_frame(np.zeros((0, 0, 3), dtype=np.uint8), TEST_CONFIG)
        self.assertEqual(result.routes, [])

    def test_wrong_dtype_returns_empty_result(self):
        frame = _line_frame(GREEN, 100).astype(np.float32) / 255.0
        result = analyze_frame(frame, TEST_CONFIG)
        self.assertEqual(result.routes, [])

    def test_input_frame_is_not_mutated(self):
        frame = _line_frame(GREEN, 100)
        snapshot = frame.copy()
        analyze_frame(frame, TEST_CONFIG)
        np.testing.assert_array_equal(frame, snapshot)

    def test_empty_config_returns_empty_result(self):
        empty_config = SceneConfig(routes=(), px_per_meter=None, min_area_px=10)
        result = analyze_frame(_line_frame(GREEN, 100), empty_config)
        self.assertEqual(result.routes, [])


class AnnotateTest(unittest.TestCase):
    def test_empty_result_keeps_shape(self):
        frame = _canvas()
        annotated = annotate(frame, SceneResult())
        self.assertEqual(annotated.shape, frame.shape)
        self.assertEqual(annotated.dtype, np.uint8)

    def test_routes_are_drawn(self):
        frame = _line_frame(GREEN, 100)
        result = analyze_frame(frame, TEST_CONFIG)
        annotated = annotate(frame, result)
        self.assertEqual(annotated.shape, frame.shape)
        self.assertFalse(np.array_equal(annotated, frame))

    def test_none_image_is_tolerated(self):
        annotated = annotate(None, SceneResult())
        self.assertEqual(annotated.ndim, 3)
        self.assertGreater(annotated.size, 0)

    def test_input_frame_is_not_mutated(self):
        frame = _line_frame(GREEN, 100)
        snapshot = frame.copy()
        annotate(frame, analyze_frame(frame, TEST_CONFIG))
        np.testing.assert_array_equal(frame, snapshot)


class SkeletonLengthTest(unittest.TestCase):
    def test_straight_horizontal_line(self):
        frame = _line_frame(GREEN, 200, y=200)
        mask = _hsv_mask(frame, (40, 120, 70), (85, 255, 255))
        length = skeleton_length_px(mask)
        # 200 px stroke; allow 10% for skeleton end effects.
        self.assertAlmostEqual(length, 200.0, delta=20.0)

    def test_empty_and_none_masks(self):
        self.assertEqual(skeleton_length_px(np.zeros((10, 10), dtype=np.uint8)), 0.0)
        self.assertEqual(skeleton_length_px(None), 0.0)


class LoadConfigTest(unittest.TestCase):
    VALID = {
        "px_per_meter": 120.0,
        "min_area_px": 250,
        "routes": [
            {
                "name": "red",
                "number": 1,
                "hsv_ranges": [
                    [[0, 120, 70], [10, 255, 255]],
                    [[170, 120, 70], [180, 255, 255]],
                ],
            },
            {
                "name": "green",
                "hsv_ranges": [[[40, 120, 70], [85, 255, 255]]],
                "dangerous": True,
            },
        ],
    }

    def _write(self, payload) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "routes.json"
        text = payload if isinstance(payload, str) else json.dumps(payload)
        path.write_text(text, encoding="utf-8")
        return path

    def test_valid_config_is_parsed(self):
        config = load_config(self._write(self.VALID))
        self.assertEqual(len(config.routes), 2)
        self.assertEqual(config.routes[0].name, "red")
        self.assertEqual(config.routes[0].number, 1)
        self.assertEqual(len(config.routes[0].hsv_ranges), 2)
        self.assertFalse(config.routes[0].dangerous)
        self.assertIsNone(config.routes[1].number)
        self.assertTrue(config.routes[1].dangerous)
        self.assertAlmostEqual(config.px_per_meter, 120.0)
        self.assertEqual(config.min_area_px, 250)

    def test_missing_file_names_path(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        missing = Path(directory.name) / "absent.json"
        with self.assertRaises(FileNotFoundError) as raised:
            load_config(missing)
        self.assertIn("absent.json", str(raised.exception))

    def test_malformed_json_raises_value_error(self):
        with self.assertRaises(ValueError):
            load_config(self._write("{not valid json"))

    def test_malformed_structure_raises_value_error(self):
        payload = {"routes": [{"name": "red", "hsv_ranges": "not-a-list"}]}
        with self.assertRaises(ValueError):
            load_config(self._write(payload))

    def test_unknown_key_raises_value_error(self):
        payload = dict(self.VALID)
        payload["hsv_range"] = []
        with self.assertRaises(ValueError):
            load_config(self._write(payload))

    def test_shipped_config_loads(self):
        config = load_config(REPO_ROOT / "config" / "routes.json")
        names = [route.name for route in config.routes]
        self.assertIn("green", names)
        dangerous = [route.name for route in config.routes if route.dangerous]
        self.assertEqual(dangerous, ["blue"])
        self.assertIsNone(config.px_per_meter)


if __name__ == "__main__":
    unittest.main()
