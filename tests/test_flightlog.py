"""Tests for the JSONL flight log."""

import json
import tempfile
import unittest
from pathlib import Path

from mars.flightlog import FlightLogger


class FlightLoggerTests(unittest.TestCase):
    def test_writes_one_json_object_per_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = FlightLogger(tmp)
            logger.log("connected", tcp="10.42.0.1:20556", dry_run=False)
            logger.log("telemetry", battery_v=8.2, state="IN_SKY")
            logger.close()

            self.assertIsNotNone(logger.path)
            lines = Path(logger.path).read_text(encoding="utf-8").splitlines()
            records = [json.loads(line) for line in lines]

        self.assertEqual([record["event"] for record in records], ["connected", "telemetry", "log_closed"])
        self.assertEqual(records[0]["tcp"], "10.42.0.1:20556")
        self.assertEqual(records[1]["state"], "IN_SKY")
        self.assertTrue(all("ts" in record for record in records))

    def test_keeps_unicode_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = FlightLogger(tmp)
            logger.log("message", text="Взлёт выполнен")
            line = Path(logger.path).read_text(encoding="utf-8").strip()
        self.assertIn("Взлёт выполнен", line)

    def test_unwritable_directory_does_not_raise(self):
        logger = FlightLogger("/proc/definitely/not/writable")
        logger.log("still", alive=True)
        self.assertIsNone(logger.path)
        self.assertIsNotNone(logger.error)


if __name__ == "__main__":
    unittest.main()
