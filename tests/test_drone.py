"""Tests for the simulated drone link and telemetry formatting."""

import tempfile
import unittest
from pathlib import Path

from mars.drone import SimulatedLink, Telemetry


class TelemetryTests(unittest.TestCase):
    def test_as_dict_rounds_and_keeps_missing_values(self):
        telemetry = Telemetry(
            state="IN_SKY",
            battery_v=8.234,
            battery_c=30.55,
            position=(1.234, -0.005, 1.5),
            altitude_m=1.48,
            yaw_deg=90.04,
        )
        data = telemetry.as_dict()
        self.assertEqual(data["state"], "IN_SKY")
        self.assertEqual(data["battery_v"], 8.23)
        self.assertEqual(data["battery_c"], 30.6)
        self.assertEqual(data["x"], 1.23)
        self.assertEqual(data["z"], 1.5)
        self.assertEqual(data["yaw_deg"], 90.0)

    def test_as_dict_handles_no_position(self):
        data = Telemetry().as_dict()
        self.assertIsNone(data["x"])
        self.assertIsNone(data["battery_v"])
        self.assertEqual(data["state"], "UNKNOWN")


class SimulatedLinkTests(unittest.TestCase):
    def test_flight_state_machine(self):
        link = SimulatedLink(frame_path=None)
        link.connect()
        self.assertEqual(link.telemetry().state, "ON_LAND")

        self.assertTrue(link.takeoff())
        self.assertEqual(link.telemetry().state, "IN_SKY")
        self.assertAlmostEqual(link.telemetry().position[2], 1.0)

        self.assertTrue(link.land())
        self.assertEqual(link.telemetry().state, "ON_LAND")
        self.assertAlmostEqual(link.telemetry().position[2], 0.0)

    def test_velocity_moves_position(self):
        link = SimulatedLink(frame_path=None)
        link.connect()
        link.takeoff()
        link.set_velocity(0.5, 0.0, 0.0, 0.0)
        link._last_update -= 0.4  # simulate 0.4 s of flight (dt is capped at 0.5)
        position = link.telemetry().position
        self.assertGreater(position[0], 0.15)
        self.assertLess(position[0], 0.25)

    def test_camera_angle_is_clamped(self):
        link = SimulatedLink(frame_path=None)
        link.set_camera_angle(-500)
        self.assertEqual(link.angle, -80.0)
        link.set_camera_angle(500)
        self.assertEqual(link.angle, 30.0)

    def test_photo_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            link = SimulatedLink(frame_path=None)
            path = link.take_photo(tmp)
            self.assertIsNotNone(path)
            self.assertTrue(Path(path).exists())
            self.assertGreater(Path(path).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
