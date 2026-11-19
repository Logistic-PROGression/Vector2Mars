"""Offline tests for the stage-1 JPEG decoder packet reader."""

from __future__ import annotations

import unittest

import decode_jpeg

SYNC = bytes.fromhex("31 6F 6B 6F")
HEADER = bytes.fromhex("01 02 03 04 05")


def _packet(offset: int, payload: bytes, sync_word: bytes = SYNC) -> str:
    body = HEADER + sync_word + offset.to_bytes(4, "little")
    body += bytes.fromhex("06 00") + payload + b"\x00" * 3 + b"\x00\x00"
    assert len(body) == decode_jpeg.PACKET_LEN
    return body.hex(" ").upper()


class ReadChunksTest(unittest.TestCase):
    def test_keeps_only_crc_ok_packets_with_valid_sync(self):
        payload = bytes(range(decode_jpeg.DATA_LEN))
        good = _packet(258, payload)  # 0x00000102, catches big-endian mistakes
        bad_crc = _packet(500, bytes(decode_jpeg.DATA_LEN))
        bad_sync = _packet(900, bytes(decode_jpeg.DATA_LEN), sync_word=b"\x00" * 4)
        text = "\n".join(
            [
                "[10:00:00] ✓ CRC OK",
                good,
                "[10:00:01] ✓ CRC FAIL",
                bad_crc,
                "[10:00:02] ✓ CRC OK",
                bad_sync,
            ]
        )

        chunks = list(decode_jpeg.read_chunks(text))

        self.assertEqual(len(chunks), 1)
        offset, chunk = chunks[0]
        self.assertEqual(offset, 258)
        self.assertEqual(chunk, payload)
        self.assertEqual(len(chunk), 54)

    def test_empty_text_yields_nothing(self):
        self.assertEqual(list(decode_jpeg.read_chunks("")), [])

    def test_non_hex_packet_is_skipped(self):
        text = "[00:00:00] ✓ CRC OK\nnot hex at all\n"
        self.assertEqual(list(decode_jpeg.read_chunks(text)), [])

    def test_wrong_packet_length_is_skipped(self):
        text = "[00:00:00] ✓ CRC OK\n" + "00 " * 10 + "\n"
        self.assertEqual(list(decode_jpeg.read_chunks(text)), [])


if __name__ == "__main__":
    unittest.main()
