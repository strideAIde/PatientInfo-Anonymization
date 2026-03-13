from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ExifTags

logger = logging.getLogger(__name__)

MIN_DIMENSION_PX: int = 64


@dataclass(frozen=True)
class LoadResult:
    image: np.ndarray
    path: Path
    original_size: tuple[int, int]


def load(path: str | Path) -> LoadResult:
    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    raw = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"cv2 could not decode image: {path}")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    h, w = img.shape[:2]
    original_size = (w, h)

    img = _apply_exif_orientation(img, path)

    h, w = img.shape[:2]
    if h < MIN_DIMENSION_PX or w < MIN_DIMENSION_PX:
        raise ValueError(f"Image is too small ({w}x{h}px) to process reliably: {path}")

    logger.debug("Loaded %s — size %dx%d", path.name, w, h)
    return LoadResult(image=img, path=path, original_size=original_size)


def _apply_exif_orientation(img: np.ndarray, path: Path) -> np.ndarray:
    try:
        pil = Image.open(path)
        exif = pil._getexif()
        if exif is None:
            return img

        orientation_key = next(
            (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None
        )
        if orientation_key is None or orientation_key not in exif:
            return img

        orientation = exif[orientation_key]
    except Exception:
        return img

    if orientation == 2:
        img = cv2.flip(img, 1)
    elif orientation == 3:
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif orientation == 4:
        img = cv2.flip(img, 0)
    elif orientation == 5:
        img = cv2.transpose(img)
        img = cv2.flip(img, 1)
    elif orientation == 6:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 7:
        img = cv2.transpose(img)
        img = cv2.flip(img, 0)
    elif orientation == 8:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return img
