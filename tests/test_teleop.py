"""Tests for the teleop control law (no curses, no hardware)."""

import unittest

from mars.teleop import (
    BACK,
    DOWN,
    FORWARD,
    KEY_COMMANDS,
    LEFT,
    RIGHT,
    STOP,
    UP,
    YAW_LEFT,
    YAW_RIGHT,
    ControlState,
)


class KeyMapTests(unittest.TestCase):
    def test_required_controls_are_mapped(self):
        for key in ("w", "s", "a", "d", "KEY_UP", "KEY_DOWN", "KEY_LEFT", "KEY_RIGHT"):
            self.assertIn(key, KEY_COMMANDS)
        self.assertEqual(KEY_COMMANDS["r"], UP)
        self.assertEqual(KEY_COMMANDS["f"], DOWN)
        self.assertEqual(KEY_COMMANDS["q"], YAW_LEFT)
        self.assertEqual(KEY_COMMANDS["e"], YAW_RIGHT)
        self.assertEqual(KEY_COMMANDS[" "], STOP)


class ControlStateTests(unittest.TestCase):
    def setUp(self):
        self.state = ControlState(speed=0.5, yaw_rate=45.0, pulse_s=0.3)

    def test_forward_pulse_expires(self):
        self.state.press(FORWARD, now=10.0)
        self.assertAlmostEqual(self.state.tick(10.1).vx, 0.5)
        self.assertTrue(self.state.tick(10.31).is_zero())

    def test_axes_combine(self):
        self.state.press(FORWARD, now=0.0)
        self.state.press(LEFT, now=0.0)
        velocity = self.state.tick(0.1)
        self.assertAlmostEqual(velocity.vx, 0.5)
        self.assertAlmostEqual(velocity.vy, 0.5)
        self.assertAlmostEqual(velocity.vz, 0.0)

    def test_opposite_key_replaces_axis(self):
        self.state.press(FORWARD, now=0.0)
        self.state.press(BACK, now=0.05)
        velocity = self.state.tick(0.1)
        self.assertAlmostEqual(velocity.vx, -0.5)

    def test_stop_clears_all_axes(self):
        self.state.press(FORWARD, now=0.0)
        self.state.press(UP, now=0.0)
        self.state.press(STOP, now=0.05)
        self.assertTrue(self.state.tick(0.06).is_zero())

    def test_speed_is_clamped(self):
        self.assertEqual(self.state.change_speed(10.0), 1.5)
        self.assertEqual(self.state.change_speed(-10.0), 0.1)

    def test_yaw_rate_is_clamped(self):
        slow = ControlState(yaw_rate=1.0)
        fast = ControlState(yaw_rate=1000.0)
        self.assertGreaterEqual(slow.yaw_rate, 10.0)
        self.assertLessEqual(fast.yaw_rate, 90.0)

    def test_unknown_command_raises(self):
        with self.assertRaises(ValueError):
            self.state.press("teleport", now=0.0)


if __name__ == "__main__":
    unittest.main()
