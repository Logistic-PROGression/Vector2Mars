"""Tests for cross-frame arrow voting (``mars.fusion``)."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from mars.arrow_routes import Arrow, routes_from_arrows
from mars.aruco import Marker
from mars.fusion import FrameObservation, fuse_routes

SIDE = 40.0
IMAGE = np.zeros((600, 800, 3), dtype=np.uint8)
# Arrow centres of a two-marker corridor; the first and last stay within the
# marker reach (3 sides) and the steps stay below the link limit (2.6 sides).
CHAIN_X = (208.0, 296.0, 384.0, 472.0)
MARKER_POINTS = {0: (120.0, 150.0), 2: (560.0, 150.0), 3: (120.0, 450.0), 4: (560.0, 450.0)}


def _marker(marker_id: int, cx: float, cy: float) -> Marker:
    half = SIDE / 2.0
    corners = np.asarray(
        [
            (cx - half, cy - half),
            (cx + half, cy - half),
            (cx + half, cy + half),
            (cx - half, cy + half),
        ],
        dtype=np.float32,
    )
    return Marker(id=marker_id, corners=corners, center=(cx, cy), side_px=SIDE)


def _markers(dx: float, dy: float) -> tuple[Marker, ...]:
    return tuple(
        _marker(marker_id, x + dx, y + dy)
        for marker_id, (x, y) in sorted(MARKER_POINTS.items())
    )


def _chain(
    dx: float,
    dy: float,
    *,
    missing: tuple[int, ...] = (),
    flip: int | None = None,
) -> list[Arrow]:
    arrows: list[Arrow] = []
    for index, x in enumerate(CHAIN_X):
        if index in missing:
            continue
        direction = (-1.0, 0.0) if index == flip else (1.0, 0.0)
        arrows.append(
            Arrow(
                center=(x + dx, 150.0 + dy),
                direction=direction,
                area_px=400 + index,
                length_px=40.0,
            )
        )
    return arrows


def _observation(
    name: str,
    dx: float,
    dy: float,
    arrows: list[Arrow],
    sharpness: float,
) -> FrameObservation:
    markers = _markers(dx, dy)
    routes = routes_from_arrows(
        arrows, list(markers), frame_shape=(IMAGE.shape[0], IMAGE.shape[1])
    )
    return FrameObservation(
        path=Path(name),
        image=IMAGE,
        markers=markers,
        arrows=tuple(arrows),
        routes=tuple(routes),
        sharpness=sharpness,
    )


class FusionTest(unittest.TestCase):
    def test_missing_arrow_in_one_frame_is_recovered_by_the_others(self) -> None:
        frames = [
            _observation("a.jpg", 0.0, 0.0, _chain(0, 0), 300.0),
            _observation("b.jpg", 18.0, -12.0, _chain(18, -12, missing=(2,)), 200.0),
            _observation("c.jpg", -14.0, 9.0, _chain(-14, 9), 100.0),
        ]
        fused = fuse_routes(frames)

        self.assertEqual(fused.mode, "fusion")
        self.assertEqual(len(fused.routes), 1)
        route = fused.routes[0]
        self.assertEqual((route.start_marker_id, route.end_marker_id), (0, 2))
        self.assertEqual(len(route.arrows), 4)
        self.assertEqual(fused.confirmed, 4)
        self.assertEqual([o.path.name for o in fused.used], ["a.jpg", "b.jpg", "c.jpg"])

    def test_phantom_arrow_supported_by_one_frame_is_dropped(self) -> None:
        phantom = Arrow(
            center=(360.0, 330.0),
            direction=(1.0, 0.0),
            area_px=500,
            length_px=40.0,
        )
        frames = [
            _observation("a.jpg", 0.0, 0.0, _chain(0, 0), 300.0),
            _observation("b.jpg", 18.0, -12.0, _chain(18, -12), 200.0),
            _observation("c.jpg", -14.0, 9.0, _chain(-14, 9) + [phantom], 100.0),
        ]
        fused = fuse_routes(frames)

        self.assertEqual(fused.mode, "fusion")
        self.assertEqual(fused.dropped, 1)
        self.assertEqual(len(fused.routes), 1)
        self.assertEqual(len(fused.routes[0].arrows), 4)

    def test_arrow_needs_a_majority_of_five_frames(self) -> None:
        frames = [
            _observation(f"{index}.jpg", shift, shift / 2, _chain(shift, shift / 2), 300.0 - index)
            for index, shift in enumerate((0.0, 10.0, -12.0, 16.0, -8.0))
        ]
        rare = Arrow(
            center=(350.0, 250.0),
            direction=(1.0, 0.0),
            area_px=500,
            length_px=40.0,
        )
        frames[1] = _observation(
            "1.jpg", 10.0, 5.0, _chain(10.0, 5.0) + [rare], 299.0
        )
        frames[3] = _observation(
            "3.jpg", 16.0, 8.0, _chain(16.0, 8.0) + [rare], 297.0
        )
        fused = fuse_routes(frames)

        self.assertEqual(fused.mode, "fusion")
        self.assertGreaterEqual(fused.dropped, 1)
        self.assertEqual(len(fused.routes), 1)
        self.assertEqual(len(fused.routes[0].arrows), 4)

    def test_flipped_head_on_one_frame_does_not_split_the_vote(self) -> None:
        frames = [
            _observation("a.jpg", 0.0, 0.0, _chain(0, 0), 300.0),
            _observation("b.jpg", 12.0, -8.0, _chain(12, -8, flip=1), 200.0),
        ]
        fused = fuse_routes(frames)

        self.assertEqual(fused.mode, "fusion")
        self.assertEqual(len(fused.routes), 1)
        route = fused.routes[0]
        self.assertEqual((route.start_marker_id, route.end_marker_id), (0, 2))
        self.assertEqual(len(route.arrows), 4)

    def test_one_frame_falls_back_to_its_own_routes(self) -> None:
        frame = _observation("only.jpg", 0.0, 0.0, _chain(0, 0), 300.0)
        fused = fuse_routes([frame])

        self.assertEqual(fused.mode, "single")
        self.assertEqual(fused.reference.path, Path("only.jpg"))
        self.assertEqual(len(fused.routes), 1)
        self.assertEqual(len(fused.routes[0].arrows), 4)

    def test_wrong_marker_position_skips_the_frame(self) -> None:
        broken = list(_markers(20.0, -10.0))
        target = next(marker for marker in broken if marker.id == 2)
        bad = _marker(2, target.center[0] + 90.0, target.center[1] + 70.0)
        broken = tuple(bad if marker.id == 2 else marker for marker in broken)
        arrows = _chain(20.0, -10.0)
        routes = routes_from_arrows(
            arrows, list(broken), frame_shape=(IMAGE.shape[0], IMAGE.shape[1])
        )
        good = _observation("good.jpg", 0.0, 0.0, _chain(0, 0), 300.0)
        suspect = FrameObservation(
            path=Path("bad.jpg"),
            image=IMAGE,
            markers=broken,
            arrows=tuple(arrows),
            routes=tuple(routes),
            sharpness=200.0,
        )
        fused = fuse_routes([good, suspect])

        self.assertEqual(fused.mode, "single")
        self.assertEqual(fused.reference.path, Path("good.jpg"))
        self.assertIn("гомография", fused.note)


if __name__ == "__main__":
    unittest.main()
