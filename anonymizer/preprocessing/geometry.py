from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import SKEW_CORRECTION_MIN_ANGLE_DEG


@dataclass(frozen=True)
class GeometryResult:
    image: np.ndarray
    transform: np.ndarray


def correct_geometry(img: np.ndarray) -> GeometryResult:
    img_rot, M_rot = _correct_skew(img)
    img_warp, M_warp = _correct_perspective(img_rot)
    transform = M_warp @ M_rot
    return GeometryResult(image=img_warp, transform=transform)


def _correct_skew(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    angle = _detect_skew_angle(img)
    if abs(angle) < SKEW_CORRECTION_MIN_ANGLE_DEG:
        return img.copy(), np.eye(3, dtype=np.float64)
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    M3 = np.vstack([M, [0, 0, 1]]).astype(np.float64)
    return rotated, M3


def _detect_skew_angle(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=80,
        minLineLength=50,
        maxLineGap=10,
    )
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        if x2 == x1:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 45:
            angles.append(angle)
    if not angles:
        return 0.0
    return float(np.median(angles))


def _correct_perspective(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    corners = _find_document_corners(img)
    if corners is None:
        return img.copy(), np.eye(3, dtype=np.float64)
    h, w = img.shape[:2]
    dst = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return warped, M.astype(np.float64)


def _find_document_corners(img: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    h, w = img.shape[:2]
    if cv2.contourArea(largest) < 0.25 * h * w:
        return None
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
    if len(approx) != 4:
        return None
    pts = approx.reshape(4, 2).astype(np.float32)
    return _order_corners(pts)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect
