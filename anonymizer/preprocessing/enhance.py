from __future__ import annotations

import cv2
import numpy as np

from config import CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID, DENOISE_H, DENOISE_H_COLOR


def enhance(img: np.ndarray) -> np.ndarray:
    img = _apply_clahe(img)
    img = _denoise(img)
    return img


def _apply_clahe(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID,
    )
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _denoise(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(
        img,
        None,
        h=DENOISE_H,
        hColor=DENOISE_H_COLOR,
        templateWindowSize=7,
        searchWindowSize=21,
    )
