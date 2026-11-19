#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Декодер JPEG из лога SDRangel.

Пакет (74 байта): 0-4 заголовок, 5-8 синхрослово, 9-12 смещение
(little-endian), 13-14 номер файла, 15-68 данные (54 байта), 72-73 CRC.

Запуск: python3 decode_jpeg.py [лог] [image.jpg]
"""

import sys
from pathlib import Path

SYNC = bytes.fromhex("31 6F 6B 6F")        # метка пакетов с изображением
DATA_START, DATA_LEN, PKT_LEN = 15, 54, 74
DEFAULT_LOGS = ["спутник бурмалдутник.txt", "map_Rx.txt", "samples/map_Rx.txt"]


def read_chunks(text):
    """Пары (смещение, данные) из пакетов с CRC OK и верным синхрословом."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "CRC OK" not in line:
            continue
        # пакет — первая непустая строка после вердикта
        hex_line = next((s.strip() for s in lines[i + 1:] if s.strip()), "")
        try:
            pkt = bytes.fromhex(hex_line)
        except ValueError:
            continue
        if len(pkt) == PKT_LEN and pkt[5:9] == SYNC:
            offset = int.from_bytes(pkt[9:13], "little")  # младший байт первым
            yield offset, pkt[DATA_START:DATA_START + DATA_LEN]


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else next(
        (Path(p) for p in DEFAULT_LOGS if Path(p).exists()), Path(DEFAULT_LOGS[0])
    )
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("image.jpg")

    buf = bytearray()  # сюда кусочки лягут по своим смещениям
    for offset, chunk in read_chunks(src.read_text(encoding="utf-8", errors="replace")):
        if len(buf) < offset + len(chunk):          # при необходимости растягиваем буфер
            buf.extend(b"\x00" * (offset + len(chunk) - len(buf)))
        buf[offset:offset + len(chunk)] = chunk     # повторные копии перезаписываются

    buf[0:2] = b"\xFF\xD8"  # возвращаем затёртый маркер начала JPEG
    dst.write_bytes(buf)
    print(f"{src} -> {dst}: {len(buf)} байт")

    try:  # необязательная проверка, если установлен Pillow
        from PIL import Image
        with Image.open(dst) as im:
            im.load()
            print(f"OK: {im.format} {im.width}x{im.height}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
