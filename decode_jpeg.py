#!/usr/bin/env python3
"""Rebuild a JPEG image from an SDRangel receiver log.

Packet layout (74 bytes): header 0-4, sync word 5-8, little-endian file
offset 9-12, file number 13-14, payload 15-68 (54 bytes), reserved 69-71,
CRC-16 72-73.

Usage: python3 decode_jpeg.py [log] [output.jpg]
"""

import sys
from pathlib import Path

SYNC = bytes.fromhex("31 6F 6B 6F")  # marks image packets
DATA_START, DATA_LEN, PACKET_LEN = 15, 54, 74
DEFAULT_LOGS = ("спутник бурмалдутник.txt", "map_Rx.txt", "samples/map_Rx.txt")


def read_chunks(text):
    """Yield (offset, payload) for every CRC OK packet with a valid sync word."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "CRC OK" not in line:
            continue
        # The packet is the first non-empty line after the verdict.
        hex_line = next((s.strip() for s in lines[i + 1:] if s.strip()), "")
        try:
            packet = bytes.fromhex(hex_line)
        except ValueError:
            continue
        if len(packet) == PACKET_LEN and packet[5:9] == SYNC:
            # Little-endian: the least significant byte comes first.
            offset = int.from_bytes(packet[9:13], "little")
            yield offset, packet[DATA_START:DATA_START + DATA_LEN]


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else next(
        (Path(p) for p in DEFAULT_LOGS if Path(p).exists()), Path(DEFAULT_LOGS[0])
    )
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("image.jpg")

    buf = bytearray()  # Holds the reassembled file.
    for offset, chunk in read_chunks(src.read_text(encoding="utf-8", errors="replace")):
        # Grow the buffer if this chunk extends past its current end.
        if len(buf) < offset + len(chunk):
            buf.extend(b"\x00" * (offset + len(chunk) - len(buf)))
        # Duplicate packets simply overwrite the same bytes.
        buf[offset:offset + len(chunk)] = chunk

    # The transmitter intentionally corrupts the SOI marker. Restore it last,
    # otherwise the packet at offset 0 would overwrite it again.
    buf[0:2] = b"\xFF\xD8"
    dst.write_bytes(buf)
    print(f"{src} -> {dst}: {len(buf)} bytes")

    try:  # Optional sanity check when Pillow is available.
        from PIL import Image
        with Image.open(dst) as img:
            img.load()
            print(f"OK: {img.format} {img.width}x{img.height}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
