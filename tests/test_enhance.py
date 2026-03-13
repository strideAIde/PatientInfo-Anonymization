import numpy as np
import pytest
import cv2

from anonymizer.preprocessing.enhance import enhance, _apply_clahe, _denoise


def _solid(value: int = 128, h: int = 200, w: int = 200) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _gradient(h: int = 200, w: int = 200) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    img[:, :, 1] = np.tile(np.linspace(0, 255, h, dtype=np.uint8).reshape(-1, 1), (1, w))
    return img


def test_enhance_returns_ndarray():
    result = enhance(_solid())
    assert isinstance(result, np.ndarray)


def test_enhance_preserves_shape():
    img = _solid(h=150, w=300)
    result = enhance(img)
    assert result.shape == img.shape


def test_enhance_preserves_dtype():
    result = enhance(_solid())
    assert result.dtype == np.uint8


def test_enhance_does_not_mutate_input():
    img = _solid(64)
    original = img.copy()
    enhance(img)
    np.testing.assert_array_equal(img, original)


def test_enhance_returns_new_array():
    img = _solid()
    result = enhance(img)
    assert result is not img


def test_clahe_output_shape_dtype():
    img = _gradient()
    result = _apply_clahe(img)
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_clahe_modifies_low_contrast_image():
    flat = _solid(128)
    result = _apply_clahe(flat)
    assert result.shape == flat.shape


def test_clahe_does_not_mutate_input():
    img = _gradient()
    original = img.copy()
    _apply_clahe(img)
    np.testing.assert_array_equal(img, original)


def test_denoise_output_shape_dtype():
    img = _gradient()
    result = _denoise(img)
    assert result.shape == img.shape
    assert result.dtype == np.uint8


def test_denoise_does_not_mutate_input():
    img = _gradient()
    original = img.copy()
    _denoise(img)
    np.testing.assert_array_equal(img, original)


def test_enhance_on_gradient_produces_valid_pixel_range():
    result = enhance(_gradient())
    assert result.min() >= 0
    assert result.max() <= 255


def test_enhance_high_contrast_image():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:100, :] = 0
    img[100:, :] = 255
    result = enhance(img)
    assert result.dtype == np.uint8
    assert result.shape == img.shape


def test_enhance_single_row_still_works():
    img = np.full((64, 64, 3), 100, dtype=np.uint8)
    result = enhance(img)
    assert result.shape == (64, 64, 3)
