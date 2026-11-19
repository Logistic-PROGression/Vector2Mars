#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сборка JPEG-изображения из лога приёмника SDRangel.

Задание: прочитать лог (по умолчанию "спутник бурмалдутник.txt"),
отобрать пакеты с CRC OK, проверить синхрослово, вытащить из каждого
пакета смещение (little-endian) и 54 байта данных, разложить кусочки
по своим местам в буфере, восстановить маркер начала JPEG FF D8
и сохранить готовый файл на диск.

Формат пакета (74 байта, нумерация с нуля):
    0..4   : заголовок            01 02 03 04 05
    5..8   : синхрослово          31 6F 6B 6F
    9..12  : смещение в файле     4 байта, little-endian
    13..14 : номер файла          06 00
    15..68 : данные               54 байта JPEG
    69..71 : резерв               00 00 00
    72..73 : CRC-16
"""

import os
import sys

# ---------------------------------------------------------------------------
# Константы протокола
# ---------------------------------------------------------------------------
SYNC = bytes.fromhex("31 6F 6B 6F")  # метка пакета с изображением (байты 5-8)
HEADER = bytes.fromhex("01 02 03 04 05")  # служебное начало пакета (байты 0-4)

PACKET_SIZE = 74        # полный размер пакета в байтах
OFFSET_START = 9        # с какого байта лежит смещение
OFFSET_LEN = 4          # длина поля смещения
DATA_START = 15         # с какого байта лежат данные
DATA_LEN = 54           # сколько байт данных в одном пакете

# Маркер начала JPEG. При передаче его специально испортили,
# поэтому после сборки его нужно вернуть на место.
JPEG_SOI = b"\xFF\xD8"


def parse_log(path):
    """
    Читает лог приёмника и возвращает список исправных пакетов (bytes).

    Логика простая: строка, содержащая "CRC OK", означает, что
    следующая непустая строка — это HEX-запись пакета. Пакеты с
    "CRC FAIL" (и любые другие строки) пропускаются.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    packets = []            # сюда складываем байты исправных пакетов
    expect_hex = False      # флаг "следующая строка — HEX-пакет"

    for line in lines:
        text = line.strip()

        # Пустые строки не влияют на состояние — просто пропускаем
        if not text:
            continue

        # Строка-вердикт приёмника
        if "CRC OK" in text:
            expect_hex = True
            continue

        # Если был CRC FAIL, то следующая строка — испорченный пакет,
        # его нужно пропустить, поэтому сбрасываем ожидание.
        if "CRC FAIL" in text:
            expect_hex = False
            continue

        # Если это не вердикт и не HEX-строка — игнорируем
        if not expect_hex:
            continue

        # Пробуем преобразовать HEX-запись в байты
        try:
            packet = bytes.fromhex(text)
        except ValueError:
            # Строка не является корректным HEX — пропускаем
            expect_hex = False
            continue

        expect_hex = False

        # На всякий случай проверяем размер пакета
        if len(packet) < PACKET_SIZE:
            continue

        packets.append(packet)

    return packets


def build_image(packets):
    """
    Собирает JPEG-файл из списка пакетов и возвращает его как bytes.
    """
    # Буфер будущего файла. bytearray позволяет менять байты по индексу.
    buffer_ = bytearray()
    used = 0          # сколько пакетов реально легло в буфер
    file_number = None  # номер файла, с которым работаем

    for packet in packets:
        # 1. Проверяем служебный заголовок (необязательно, но полезно)
        if packet[0:5] != HEADER:
            continue

        # 2. Проверяем синхрослово: пакет должен относиться к изображению
        if packet[5:9] != SYNC:
            continue

        # 3. Номер файла (байты 13-14, little-endian).
        #    Если по каналу идёт несколько файлов, берём первый
        #    встреченный номер и собираем только его пакеты.
        pkt_file_number = int.from_bytes(packet[13:15], "little")
        if file_number is None:
            file_number = pkt_file_number
        elif pkt_file_number != file_number:
            continue

        # 4. Смещение — 4 байта little-endian (младший байт первым!)
        #    Это место, куда нужно положить кусочек данных.
        offset = int.from_bytes(
            packet[OFFSET_START:OFFSET_START + OFFSET_LEN], "little"
        )

        # 5. Сами данные изображения
        chunk = packet[DATA_START:DATA_START + DATA_LEN]

        # 6. Увеличиваем буфер, если кусочек выходит за его границы
        end = offset + len(chunk)
        if end > len(buffer_):
            buffer_.extend(b"\x00" * (end - len(buffer_)))

        # 7. Кладём кусочек на своё место.
        #    Повторно принятый пакет просто перезапишет тот же участок —
        #    у исправных копий данные одинаковые.
        buffer_[offset:end] = chunk
        used += 1

    if used == 0:
        raise RuntimeError(
            "В логе не найдено ни одного исправного пакета с изображением."
        )

    # 8. Восстанавливаем маркер начала JPEG FF D8.
    #    При передаче первые два байта были специально изменены.
    if len(buffer_) < 2:
        raise RuntimeError("Собранных данных слишком мало для JPEG.")
    buffer_[0:2] = JPEG_SOI

    print(f"Использовано пакетов: {used}")
    print(f"Номер файла: {file_number}")
    print(f"Размер собранного файла: {len(buffer_)} байт")

    return bytes(buffer_)


def main():
    # Путь к логу: первый аргумент командной строки или файл по умолчанию.
    # Скрипт лежит рядом с логом, поэтому по умолчанию берём файл из
    # папки скрипта.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(base_dir, "спутник бурмалдутник.txt")
    )
    out_path = (
        sys.argv[2]
        if len(sys.argv) > 2
        else os.path.join(base_dir, "image.jpg")
    )

    print(f"Читаю лог: {log_path}")
    packets = parse_log(log_path)
    print(f"Найдено исправных пакетов (CRC OK): {len(packets)}")

    image = build_image(packets)

    # Записываем результат в двоичном режиме
    with open(out_path, "wb") as f:
        f.write(image)
    print(f"Изображение сохранено: {out_path}")

    # Необязательная проверка: если установлен Pillow, убедимся,
    # что картинка действительно читается, и выведем её размеры.
    try:
        from PIL import Image

        with Image.open(out_path) as img:
            img.load()  # полное декодирование
            print(f"Проверка OK: {img.format}, размер {img.size[0]}x{img.size[1]}")
    except ImportError:
        print("Pillow не установлен — проверку картинки пропускаю.")
    except Exception as exc:  # noqa: BLE001
        print(f"Внимание: файл записан, но не читается как изображение: {exc}")


if __name__ == "__main__":
    main()
