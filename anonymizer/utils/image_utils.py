from __future__ import annotations

import numpy as np

from config import BLUR_PADDING_PX


def pad_bbox(
    bbox: tuple[int, int, int, int],
    img_shape: tuple[int, ...],
    padding: int = BLUR_PADDING_PX,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    h, w = img_shape[:2]
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(w - 1, x2 + padding),
        min(h - 1, y2 + padding),
    )


def map_bbox_to_original(
    bbox: tuple[int, int, int, int],
    transform: np.ndarray,
    original_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    inv = np.linalg.inv(transform)
    x1, y1, x2, y2 = bbox
    corners = np.array(
        [[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]],
        dtype=np.float64,
    )
    mapped = (inv @ corners.T).T
    mapped = mapped[:, :2] / mapped[:, 2:3]

    h, w = original_shape[:2]
    ox1 = int(np.clip(np.floor(mapped[:, 0].min()), 0, w - 1))
    oy1 = int(np.clip(np.floor(mapped[:, 1].min()), 0, h - 1))
    ox2 = int(np.clip(np.ceil(mapped[:, 0].max()), 0, w - 1))
    oy2 = int(np.clip(np.ceil(mapped[:, 1].max()), 0, h - 1))
    return (ox1, oy1, ox2, oy2)
