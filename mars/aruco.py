"""ArUco marker detection on camera frames and satellite images.

Low-level detector shared by the stage-2 pipeline: :mod:`mars.scale` uses it to
estimate pixels per metre (task B2) and :mod:`mars.numbering` can use marker
ids to label routes (task A2). Detection is defensive: unsupported frames yield
an empty list instead of raising, so a bad camera frame never stops the flight
program.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

DEFAULT_DICTIONARY = "DICT_4X4_50"


@dataclass(frozen=True)
class Marker:
    """A single ArUco marker in image coordinates."""

    id: int
    corners: np.ndarray = field(repr=False, compare=False)  # (4, 2), clockwise
    center: tuple[float, float]
    side_px: float


def detect_markers(
    image: object, dictionary: str = DEFAULT_DICTIONARY
) -> list[Marker]:
    """Return markers found in ``image``, sorted by id.

    ``image`` may be BGR, BGRA or grayscale ``uint8``; anything else yields an
    empty list. ``dictionary`` is any predefined OpenCV name such as
    ``DICT_4X4_50``; an unknown name raises :class:`ValueError`.
    """
    gray = _as_gray(image)
    if gray is None:
        return []

    aruco_dict = cv2.aruco.getPredefinedDictionary(_dictionary_id(dictionary))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    try:
        corners, ids, _ = detector.detectMarkers(gray)
    except cv2.error:
        return []
    if ids is None:
        return []

    markers: list[Marker] = []
    for marker_id, quad in zip(ids.ravel(), corners):
        points = np.asarray(quad, dtype=np.float32).reshape(4, 2)
        markers.append(
            Marker(
                id=int(marker_id),
                corners=points,
                center=(float(points[:, 0].mean()), float(points[:, 1].mean())),
                side_px=_side_px(points),
            )
        )
    return sorted(markers, key=lambda marker: marker.id)


def annotate_markers(image: object, markers: list[Marker] | None) -> np.ndarray:
    """Return a debug copy of ``image`` with marker outlines, ids and centres."""
    base = _as_bgr(image)
    if base is None:
        base = np.zeros((480, 640, 3), dtype=np.uint8)
    canvas = base.copy()
    height, width = canvas.shape[:2]
    thickness = max(2, min(4, round(min(height, width) / 200)))

    for marker in markers or ():
        quad = marker.corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [quad], True, (0, 255, 0), thickness)
        x, y = (int(round(value)) for value in marker.center)
        cv2.circle(canvas, (x, y), max(2, thickness), (0, 0, 255), -1)
        _draw_label(
            canvas,
            f"id={marker.id}",
            (x + 2 * thickness, y - 2 * thickness),
            (0, 255, 0),
        )
    return canvas


def _draw_label(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    """Draw a marker id on a dark plate so it stays readable on any background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.5, 1
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, scale, thickness
    )
    max_x = max(0, canvas.shape[1] - text_width - 8)
    max_y = max(text_height + 4, canvas.shape[0] - baseline - 4)
    x = min(max(0, origin[0]), max_x)
    y = min(max(text_height + 4, origin[1]), max_y)

    cv2.rectangle(
        canvas,
        (x - 3, y - text_height - 3),
        (x + text_width + 3, y + baseline + 3),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        canvas, text, (x, y), font, scale, color, thickness, cv2.LINE_AA
    )


def _side_px(points: np.ndarray) -> float:
    """Mean edge length of the marker quad; used for scale estimation."""
    edges = [
        float(np.linalg.norm(points[(i + 1) % 4] - points[i])) for i in range(4)
    ]
    return float(np.mean(edges))


def _dictionary_id(name: str) -> int:
    if not isinstance(name, str) or not name.startswith("DICT_"):
        raise ValueError(f"unknown ArUco dictionary: {name!r}")
    value = getattr(cv2.aruco, name, None)
    if value is None:
        raise ValueError(f"unknown ArUco dictionary: {name!r}")
    return value


def _as_gray(image: object) -> np.ndarray | None:
    """Return a grayscale ``uint8`` view/copy of a supported image, or ``None``."""
    if image is None:
        return None
    if not isinstance(image, np.ndarray):
        try:
            image = np.asarray(image)
        except Exception:
            return None
    if not isinstance(image, np.ndarray) or image.size == 0:
        return None
    if image.dtype != np.uint8 or image.ndim not in (2, 3):
        return None
    if image.ndim == 2:
        return image
    channels = image.shape[2]
    if channels == 1:
        return image[:, :, 0]
    if channels == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return None


def _as_bgr(image: object) -> np.ndarray | None:
    """Return a BGR ``uint8`` copy of a supported image, or ``None``."""
    if image is None:
        return None
    if not isinstance(image, np.ndarray):
        try:
            image = np.asarray(image)
        except Exception:
            return None
    if not isinstance(image, np.ndarray) or image.size == 0:
        return None
    if image.dtype != np.uint8 or image.ndim not in (2, 3):
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    channels = image.shape[2]
    if channels == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    if channels == 3:
        return image.copy()
    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return None


def main(argv: list[str] | None = None) -> int:
    """CLI: detect markers in an image and optionally save a debug overlay."""
    parser = argparse.ArgumentParser(description="Найти ArUco-метки на изображении.")
    parser.add_argument("image", help="путь к изображению (JPEG/PNG)")
    parser.add_argument(
        "--dict",
        default=DEFAULT_DICTIONARY,
        dest="dictionary",
        help=f"словарь OpenCV (по умолчанию {DEFAULT_DICTIONARY})",
    )
    parser.add_argument(
        "--marker-size-m",
        type=float,
        default=None,
        help="физический размер стороны маркера в метрах (для масштаба)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="сохранить изображение с разметкой по указанному пути",
    )
    args = parser.parse_args(argv)

    try:
        image = cv2.imread(str(args.image))
    except cv2.error:
        image = None
    if image is None:
        print(f"Не удалось прочитать изображение: {args.image}", file=sys.stderr)
        return 2

    try:
        markers = detect_markers(image, args.dictionary)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    print(f"Найдено меток: {len(markers)}")
    for marker in markers:
        print(
            f"  id={marker.id}  "
            f"центр=({marker.center[0]:.1f}, {marker.center[1]:.1f})  "
            f"сторона={marker.side_px:.1f} px"
        )

    if args.marker_size_m is not None:
        from mars.scale import px_per_meter

        try:
            scale = px_per_meter(markers, args.marker_size_m)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2
        if scale is None:
            print("Недостаточно меток для оценки масштаба.")
        else:
            print(f"Масштаб: {scale:.1f} px/м")

    if args.out:
        path = Path(args.out)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(path), annotate_markers(image, markers)):
                print(f"Не удалось сохранить разметку: {path}", file=sys.stderr)
                return 2
        except OSError as exc:
            print(f"Не удалось сохранить разметку: {path}: {exc}", file=sys.stderr)
            return 2
        print(f"Разметка сохранена: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
