from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import BLUR_KERNEL_SIZE, BLUR_PASSES
from anonymizer.utils.image_utils import map_bbox_to_original, pad_bbox


@dataclass(frozen=True)
class RedactResult:
    image: np.ndarray


def redact(
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    transform: np.ndarray,
) -> RedactResult:
    out = image.copy()
    for box in boxes:
        original_box = map_bbox_to_original(box, transform, image.shape)
        padded_box = pad_bbox(original_box, image.shape)
        _apply_blur(out, padded_box)
    return RedactResult(image=out)


def _apply_blur(img: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = bbox
    roi = img[y1 : y2 + 1, x1 : x2 + 1]
    k = BLUR_KERNEL_SIZE
    for _ in range(BLUR_PASSES):
        roi = cv2.GaussianBlur(roi, (k, k), 0)
    img[y1 : y2 + 1, x1 : x2 + 1] = roi
