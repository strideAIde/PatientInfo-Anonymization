"""
Tests for anonymizer.preprocessing.loader
==========================================
All tests use synthetic images generated in-memory — no real patient data.
"""

import struct
import zlib
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from anonymizer.preprocessing.loader import LoadResult, load, MIN_DIMENSION_PX


# ── Helpers ────────────────────────────────────────────────────────────────────

def _save_bgr(path: Path, width: int = 200, height: int = 150, channels: int = 3) -> Path:
    """Write a synthetic solid-colour BGR image to *path* as JPEG."""
    img = np.full((height, width, channels), 128, dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


def _save_gray(path: Path, width: int = 200, height: int = 150) -> Path:
    """Write a grayscale PNG."""
    img = np.full((height, width), 200, dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


def _save_rgba(path: Path, width: int = 200, height: int = 150) -> Path:
    """Write a 4-channel BGRA PNG."""
    img = np.full((height, width, 4), 128, dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


# ── Basic loading ──────────────────────────────────────────────────────────────

def test_load_returns_load_result(tmp_path):
    p = _save_bgr(tmp_path / "sample.jpg")
    result = load(p)
    assert isinstance(result, LoadResult)


def test_load_image_is_uint8_bgr(tmp_path):
    p = _save_bgr(tmp_path / "sample.jpg")
    result = load(p)
    assert result.image.dtype == np.uint8
    assert result.image.ndim == 3
    assert result.image.shape[2] == 3  # BGR


def test_load_correct_dimensions(tmp_path):
    p = _save_bgr(tmp_path / "sample.jpg", width=320, height=240)
    result = load(p)
    h, w = result.image.shape[:2]
    assert w == 320
    assert h == 240


def test_load_stores_path(tmp_path):
    p = _save_bgr(tmp_path / "sample.jpg")
    result = load(p)
    assert result.path == p.resolve()


def test_load_stores_original_size(tmp_path):
    p = _save_bgr(tmp_path / "sample.jpg", width=640, height=480)
    result = load(p)
    assert result.original_size == (640, 480)


def test_load_accepts_string_path(tmp_path):
    p = _save_bgr(tmp_path / "sample.jpg")
    result = load(str(p))          # pass str, not Path
    assert result.image is not None


# ── Channel normalisation ──────────────────────────────────────────────────────

def test_grayscale_converted_to_bgr(tmp_path):
    p = _save_gray(tmp_path / "gray.png")
    result = load(p)
    assert result.image.ndim == 3
    assert result.image.shape[2] == 3


def test_rgba_converted_to_bgr(tmp_path):
    p = _save_rgba(tmp_path / "rgba.png")
    result = load(p)
    assert result.image.shape[2] == 3


# ── Error cases ────────────────────────────────────────────────────────────────

def test_file_not_found_raises():
    with pytest.raises(FileNotFoundError):
        load("/nonexistent/path/image.jpg")


def test_corrupt_file_raises(tmp_path):
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"\x00\x01\x02\x03 not an image")
    with pytest.raises(ValueError, match="could not decode"):
        load(corrupt)


def test_too_small_raises(tmp_path):
    tiny = tmp_path / "tiny.jpg"
    img = np.full((10, 10, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(tiny), img)
    with pytest.raises(ValueError, match="too small"):
        load(tiny)


def test_exactly_min_dimension_is_accepted(tmp_path):
    """Image at exactly the minimum size should not raise."""
    p = tmp_path / "minsize.jpg"
    img = np.full((MIN_DIMENSION_PX, MIN_DIMENSION_PX, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(p), img)
    result = load(p)
    assert result.image.shape[:2] == (MIN_DIMENSION_PX, MIN_DIMENSION_PX)


# ── EXIF orientation ───────────────────────────────────────────────────────────

def _write_jpeg_with_exif_orientation(path: Path, orientation: int, width=200, height=100):
    """Create a non-square JPEG with a specific EXIF orientation tag."""
    img = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
    # Build a minimal EXIF blob with the Orientation tag (0x0112)
    exif_bytes = _make_exif_orientation(orientation)
    img.save(str(path), format="JPEG", exif=exif_bytes)


def _make_exif_orientation(orientation: int) -> bytes:
    """Build a minimal EXIF blob with only the Orientation tag."""
    # Minimal EXIF: Exif header + IFD with 1 entry (Orientation)
    # Big-endian TIFF
    ifd_entry = struct.pack(">HHII", 0x0112, 3, 1, orientation << 16)
    ifd = struct.pack(">H", 1) + ifd_entry + struct.pack(">I", 0)
    tiff = b"MM\x00\x2a" + struct.pack(">I", 8) + ifd
    return b"Exif\x00\x00" + tiff


def test_exif_orientation_1_unchanged(tmp_path):
    """Orientation=1 (normal) — image dimensions should not change."""
    p = tmp_path / "orient1.jpg"
    _write_jpeg_with_exif_orientation(p, orientation=1, width=200, height=100)
    result = load(p)
    h, w = result.image.shape[:2]
    assert w == 200 and h == 100


def test_exif_orientation_6_rotates_90cw(tmp_path):
    """Orientation=6 (90° CW rotation needed) — width and height swap."""
    p = tmp_path / "orient6.jpg"
    _write_jpeg_with_exif_orientation(p, orientation=6, width=200, height=100)
    result = load(p)
    h, w = result.image.shape[:2]
    # After 90° CW rotation, original 200w×100h becomes 100w×200h
    assert w == 100 and h == 200


def test_exif_orientation_3_rotates_180(tmp_path):
    """Orientation=3 (180° rotation) — dimensions unchanged, content rotated."""
    p = tmp_path / "orient3.jpg"
    _write_jpeg_with_exif_orientation(p, orientation=3, width=200, height=100)
    result = load(p)
    h, w = result.image.shape[:2]
    assert w == 200 and h == 100


def test_exif_orientation_8_rotates_90ccw(tmp_path):
    """Orientation=8 (90° CCW rotation needed) — width and height swap."""
    p = tmp_path / "orient8.jpg"
    _write_jpeg_with_exif_orientation(p, orientation=8, width=200, height=100)
    result = load(p)
    h, w = result.image.shape[:2]
    assert w == 100 and h == 200
