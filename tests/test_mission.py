"""Tests for the automated photo mission, driven by a virtual clock."""

import json
import math
import tempfile
import unittest
from pathlib import Path

from mars.drone import DroneUnavailable, Telemetry
from mars.flightlog import FlightLogger
from mars.mission import (
    MissionConfig,
    height_above_ground,
    run_photo_mission,
)


class FakeClock:
    """Virtual monotonic clock: ``sleep`` only moves time forward."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(0.0, float(seconds))


class FakeLink:
    """Minimal DroneLink stand-in with integrable altitude."""

    def __init__(
        self,
        clock: FakeClock,
        *,
        photo_failures: int = 0,
        tilt_error: Exception | None = None,
        can_move: bool = True,
        can_translate: bool = True,
        interrupt_on_velocity: bool = False,
        yaw_deg: float = 0.0,
    ) -> None:
        self.clock = clock
        self.photo_failures = photo_failures
        self.tilt_error = tilt_error
        self.can_move = can_move
        self.can_translate = can_translate
        self.interrupt_on_velocity = interrupt_on_velocity
        self.yaw_deg = yaw_deg
        self.state = "ON_LAND"
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.angle = None
        self.land_calls = 0
        self.photo_calls = 0
        self.photos: list[Path] = []
        self.camera_error = None
        self._last = clock.now

    def telemetry(self) -> Telemetry:
        now = self.clock.now
        dt = max(0.0, now - self._last)
        self._last = now
        if self.can_move:
            self.z = max(0.0, self.z + self.vz * dt)
            if self.can_translate:
                yaw = math.radians(self.yaw_deg)
                self.x += (self.vx * math.cos(yaw) - self.vy * math.sin(yaw)) * dt
                self.y += (self.vx * math.sin(yaw) + self.vy * math.cos(yaw)) * dt
        return Telemetry(
            state=self.state,
            battery_v=8.2,
            position=(self.x, self.y, self.z),
            altitude_m=self.z,
            yaw_deg=self.yaw_deg,
        )

    def takeoff(self) -> bool:
        self.state = "IN_SKY"
        self.z = 1.0
        self._last = self.clock.now
        return True

    def land(self) -> bool:
        self.state = "ON_LAND"
        self.z = 0.0
        self.vx = self.vy = self.vz = 0.0
        self.land_calls += 1
        return True

    def stop(self) -> None:
        self.vx = self.vy = self.vz = 0.0

    def set_velocity(self, vx, vy, vz, yaw_rate_deg, interval=0.2) -> None:
        if self.interrupt_on_velocity:
            raise KeyboardInterrupt
        self.vx = float(vx)
        self.vy = float(vy)
        self.vz = float(vz)

    def set_camera_angle(self, angle, priority="high", attempts=3) -> bool:
        if self.tilt_error is not None:
            raise self.tilt_error
        self.angle = float(angle)
        return True

    def take_photo(self, directory="photos") -> Path | None:
        self.photo_calls += 1
        if self.photo_calls <= self.photo_failures:
            return None
        path = Path(directory) / f"photo_{self.photo_calls}.jpg"
        self.photos.append(path)
        return path


def _run(config, **link_kwargs):
    with tempfile.TemporaryDirectory() as tmp:
        clock = FakeClock()
        link = FakeLink(clock, **link_kwargs)
        logger = FlightLogger(tmp)
        result = run_photo_mission(
            link,
            config,
            logger,
            Path(tmp) / "photos",
            notice=lambda _text: None,
            clock=clock,
            sleep=clock.sleep,
        )
        self_path = logger.path
        records = [
            json.loads(line)
            for line in Path(self_path).read_text(encoding="utf-8").splitlines()
        ]
        logger.close()
    return result, records, link, clock


def _events(records):
    return [record["event"] for record in records]


class PhotoMissionTests(unittest.TestCase):
    def test_full_mission_climbs_photos_and_lands(self):
        config = MissionConfig(altitude_m=2.7, settle_s=0.5, tolerance_m=0.05)
        result, records, link, _clock = _run(config)

        self.assertTrue(result.ok)
        self.assertIsNotNone(result.photo)
        self.assertTrue(result.photo.name.endswith(".jpg"))
        self.assertEqual(result.photo_attempts, 1)
        self.assertAlmostEqual(result.climbed_m, 2.7, delta=0.06)
        self.assertEqual(link.state, "ON_LAND")
        self.assertEqual(link.land_calls, 1)
        self.assertAlmostEqual(link.x, 1.3, delta=0.1)
        self.assertAlmostEqual(link.y, 1.3, delta=0.1)
        self.assertAlmostEqual(link.angle, -80.0)  # physical -90 → SDK -80

        events = _events(records)
        start = records[events.index("mission_start")]
        self.assertEqual(start["altitude_m"], 2.7)
        for expected in (
            "mission_start",
            "ground_reference",
            "tilt",
            "takeoff",
            "takeoff_reference",
            "climb_done",
            "move_done",
            "photo",
            "descent_done",
            "land",
            "mission_complete",
        ):
            self.assertIn(expected, events)
        self.assertLess(events.index("tilt"), events.index("takeoff"))
        self.assertLess(events.index("climb_done"), events.index("move_done"))
        self.assertLess(events.index("move_done"), events.index("photo"))
        self.assertLess(events.index("photo"), events.index("land"))
        final = records[events.index("mission_complete")]
        self.assertTrue(final["ok"])
        self.assertEqual(final["reason"], "")

    def test_moves_to_target_with_rotated_yaw(self):
        config = MissionConfig(
            altitude_m=1.5,
            settle_s=0.0,
            target_x_m=1.0,
            target_y_m=0.5,
        )
        result, records, link, _clock = _run(config, yaw_deg=90.0)

        self.assertTrue(result.ok)
        self.assertAlmostEqual(link.x, 1.0, delta=0.1)
        self.assertAlmostEqual(link.y, 0.5, delta=0.1)
        self.assertIn("move_done", _events(records))

    def test_move_timeout_lands_automatically(self):
        config = MissionConfig(
            altitude_m=1.5,
            settle_s=0.0,
            move_timeout_s=1.0,
        )
        result, records, link, _clock = _run(config, can_translate=False)

        self.assertFalse(result.ok)
        self.assertIn("не достигнута", result.reason)
        self.assertEqual(link.state, "ON_LAND")
        self.assertEqual(link.land_calls, 1)
        self.assertIn("auto_land", _events(records))

    def test_photo_retries_exhaust_and_drone_still_lands(self):
        config = MissionConfig(
            altitude_m=1.5, settle_s=0.0, photo_retries=3, photo_delay_s=0.2
        )
        result, records, link, _clock = _run(config, photo_failures=3)

        self.assertFalse(result.ok)
        self.assertIsNone(result.photo)
        self.assertEqual(result.photo_attempts, 3)
        self.assertEqual(link.photo_calls, 3)
        self.assertEqual(link.state, "ON_LAND")
        self.assertEqual(link.land_calls, 1)
        self.assertEqual(result.reason, "фото не получено")
        self.assertEqual(_events(records).count("photo_failed"), 3)

    def test_unavailable_servo_does_not_abort_mission(self):
        config = MissionConfig(altitude_m=1.5, settle_s=0.0)
        result, records, link, _clock = _run(
            config, tilt_error=DroneUnavailable("только на борту")
        )

        self.assertTrue(result.ok)
        self.assertIsNone(link.angle)
        self.assertIn("tilt_unavailable", _events(records))

    def test_climb_timeout_lands_automatically(self):
        config = MissionConfig(
            altitude_m=2.7, settle_s=0.0, climb_timeout_s=1.0
        )
        result, records, link, _clock = _run(config, can_move=False)

        self.assertFalse(result.ok)
        self.assertIn("не достигнута", result.reason)
        self.assertEqual(link.state, "ON_LAND")
        self.assertEqual(link.land_calls, 1)
        self.assertIn("mission_error", _events(records))
        self.assertIn("auto_land", _events(records))

    def test_interrupt_lands_and_reports(self):
        config = MissionConfig(altitude_m=2.7, settle_s=0.0)
        result, records, link, _clock = _run(config, interrupt_on_velocity=True)

        self.assertTrue(result.interrupted)
        self.assertFalse(result.ok)
        self.assertEqual(link.state, "ON_LAND")
        self.assertEqual(link.land_calls, 1)
        self.assertIn("mission_interrupted", _events(records))


class MissionConfigTests(unittest.TestCase):
    def test_rejects_bad_altitude(self):
        with self.assertRaises(ValueError):
            MissionConfig(altitude_m=0)
        with self.assertRaises(ValueError):
            MissionConfig(altitude_m=4.0, max_altitude_m=3.0)
        with self.assertRaises(ValueError):
            MissionConfig(tilt_deg=-100)

    def test_rejects_bad_retries_and_landing_height(self):
        with self.assertRaises(ValueError):
            MissionConfig(photo_retries=0)
        with self.assertRaises(ValueError):
            MissionConfig(altitude_m=2.7, landing_height_m=3.0)

    def test_rejects_bad_move_settings(self):
        with self.assertRaises(ValueError):
            MissionConfig(move_speed_ms=0)
        with self.assertRaises(ValueError):
            MissionConfig(move_tolerance_m=0)
        with self.assertRaises(ValueError):
            MissionConfig(move_timeout_s=0)
        with self.assertRaises(ValueError):
            MissionConfig(target_x_m=float("nan"))


class HeightTests(unittest.TestCase):
    def test_prefers_lps_position(self):
        telemetry = Telemetry(position=(0.0, 0.0, 2.2), altitude_m=1.0)
        baseline = Telemetry(position=(0.0, 0.0, 0.5), altitude_m=100.0)
        self.assertAlmostEqual(height_above_ground(telemetry, baseline), 1.7)

    def test_falls_back_to_barometric_altitude(self):
        telemetry = Telemetry(position=None, altitude_m=12.0)
        baseline = Telemetry(position=None, altitude_m=10.0)
        self.assertAlmostEqual(height_above_ground(telemetry, baseline), 2.0)

    def test_returns_none_without_any_source(self):
        self.assertIsNone(height_above_ground(Telemetry(), Telemetry()))


if __name__ == "__main__":
    unittest.main()
