"""Tests for the photo-mission CLI defaults and the arrow hint."""

from __future__ import annotations

import unittest

import photo_mission
from photo_mission import (
    CODE_DEFAULTS,
    DEFAULT_EXPECTED_ARROWS,
    CodeDefaults,
    _expected_arrows,
    _parse_args,
)


class ExpectedArrowsTest(unittest.TestCase):
    def test_disabled_hint_stays_disabled(self) -> None:
        self.assertIsNone(_expected_arrows(None))

    def test_flag_without_numbers_uses_the_known_counts(self) -> None:
        args = _parse_args(["--expect-arrows"])

        self.assertEqual(args.expect_arrows, [])
        self.assertEqual(
            _expected_arrows(args.expect_arrows), DEFAULT_EXPECTED_ARROWS
        )
        self.assertEqual(DEFAULT_EXPECTED_ARROWS, (13, 8, 3))

    def test_flag_accepts_other_counts(self) -> None:
        args = _parse_args(["--expect-arrows", "5", "2", "9"])

        self.assertEqual(_expected_arrows(args.expect_arrows), (5, 2, 9))


class CodeDefaultsTest(unittest.TestCase):
    def test_flags_use_code_defaults(self) -> None:
        args = _parse_args([])

        self.assertEqual(args.local, CODE_DEFAULTS.local)
        self.assertEqual(args.no_camera, CODE_DEFAULTS.no_camera)
        self.assertEqual(args.dry_run, CODE_DEFAULTS.dry_run)
        self.assertEqual(args.annotate_all, CODE_DEFAULTS.annotate_all)
        self.assertEqual(args.turn_photos, CODE_DEFAULTS.turn_photos)
        self.assertEqual(
            args.expect_arrows,
            None
            if CODE_DEFAULTS.expect_arrows is None
            else list(CODE_DEFAULTS.expect_arrows),
        )

    def test_hardcoded_defaults_enable_flags_without_arguments(self) -> None:
        original = photo_mission.CODE_DEFAULTS
        try:
            photo_mission.CODE_DEFAULTS = CodeDefaults(
                annotate_all=True,
                expect_arrows=(13, 8, 3),
            )
            args = _parse_args([])
        finally:
            photo_mission.CODE_DEFAULTS = original

        self.assertTrue(args.annotate_all)
        self.assertEqual(args.expect_arrows, [13, 8, 3])


class TurnPhotoFlagsTest(unittest.TestCase):
    def test_turn_photos_are_on_by_default(self) -> None:
        args = _parse_args([])

        self.assertTrue(args.turn_photos)
        self.assertEqual(args.turn_angles, [0.0, -45.0, -90.0])

    def test_no_turn_photos_turns_them_off(self) -> None:
        args = _parse_args(["--no-turn-photos"])

        self.assertFalse(args.turn_photos)
        self.assertEqual(args.turn_angles, [0.0, -45.0, -90.0])

    def test_turn_angles_and_yaw_limits_are_parsed(self) -> None:
        args = _parse_args(
            [
                "--turn-angles", "0", "-90",
                "--yaw-speed", "20",
                "--yaw-tolerance", "5",
                "--yaw-settle", "0",
                "--yaw-timeout", "10",
            ]
        )

        self.assertEqual(args.turn_angles, [0.0, -90.0])
        self.assertAlmostEqual(args.yaw_speed, 20.0)
        self.assertAlmostEqual(args.yaw_tolerance, 5.0)
        self.assertAlmostEqual(args.yaw_settle, 0.0)
        self.assertAlmostEqual(args.yaw_timeout, 10.0)

    def test_code_defaults_can_disable_turns(self) -> None:
        original = photo_mission.CODE_DEFAULTS
        try:
            photo_mission.CODE_DEFAULTS = CodeDefaults(turn_photos=False)
            args = _parse_args([])
        finally:
            photo_mission.CODE_DEFAULTS = original

        self.assertFalse(args.turn_photos)


if __name__ == "__main__":
    unittest.main()
