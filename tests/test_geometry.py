import cv2
import numpy as np
import pytest

from anonymizer.preprocessing.geometry import (
    GeometryResult,
    correct_geometry,
    _correct_skew,
    _detect_skew_angle,
    _correct_perspective,
    _find_document_corners,
    _order_corners,
)


def _solid(h: int = 300, w: int = 400, value: int = 200) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _image_with_horizontal_lines(h: int = 300, w: int = 400) -> np.ndarray:
    img = np.full((h, w, 3), 240, dtype=np.uint8)
    for y in range(30, h, 40):
        cv2.line(img, (10, y), (w - 10, y), (0, 0, 0), 2)
    return img


def _image_rotated(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def test_correct_geometry_returns_geometry_result():
    result = correct_geometry(_solid())
    assert isinstance(result, GeometryResult)


def test_correct_geometry_output_shape_matches_input():
    img = _solid(h=200, w=300)
    result = correct_geometry(img)
    assert result.image.shape == img.shape


def test_correct_geometry_output_dtype_uint8():
    result = correct_geometry(_solid())
    assert result.image.dtype == np.uint8


def test_correct_geometry_transform_is_3x3_float64():
    result = correct_geometry(_solid())
    assert result.transform.shape == (3, 3)
    assert result.transform.dtype == np.float64


def test_correct_geometry_does_not_mutate_input():
    img = _solid()
    original = img.copy()
    correct_geometry(img)
    np.testing.assert_array_equal(img, original)


def test_correct_geometry_returns_new_array():
    img = _solid()
    result = correct_geometry(img)
    assert result.image is not img


def test_skew_angle_near_zero_for_horizontal_lines():
    img = _image_with_horizontal_lines()
    angle = _detect_skew_angle(img)
    assert abs(angle) < 5.0


def test_skew_angle_nonzero_for_rotated_image():
    img = _image_with_horizontal_lines()
    rotated = _image_rotated(img, -8.0)
    angle = _detect_skew_angle(rotated)
    assert abs(angle) > 1.0


def test_correct_skew_identity_for_small_angle():
    img = _image_with_horizontal_lines()
    _, M = _correct_skew(img)
    np.testing.assert_array_almost_equal(M, np.eye(3), decimal=5)


def test_correct_skew_nonidentity_transform_for_skewed_image():
    img = _image_with_horizontal_lines()
    rotated = _image_rotated(img, -8.0)
    _, M = _correct_skew(rotated)
    assert not np.allclose(M, np.eye(3), atol=1e-3)


def test_correct_skew_output_shape_matches_input():
    img = _image_with_horizontal_lines()
    rotated = _image_rotated(img, -5.0)
    out, _ = _correct_skew(rotated)
    assert out.shape == rotated.shape


def test_detect_skew_returns_zero_for_solid_image():
    angle = _detect_skew_angle(_solid())
    assert angle == 0.0


def test_correct_perspective_identity_for_solid_image():
    img = _solid()
    out, M = _correct_perspective(img)
    assert out.shape == img.shape
    np.testing.assert_array_almost_equal(M, np.eye(3), decimal=5)


def test_find_document_corners_returns_none_for_noisy_image():
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, (300, 400, 3), dtype=np.uint8)
    result = _find_document_corners(noise)
    assert result is None


def test_order_corners_top_left_has_smallest_sum():
    pts = np.array([[100, 200], [300, 50], [350, 280], [50, 320]], dtype=np.float32)
    ordered = _order_corners(pts)
    sums = ordered.sum(axis=1)
    assert np.argmin(sums) == 0


def test_order_corners_bottom_right_has_largest_sum():
    pts = np.array([[100, 200], [300, 50], [350, 280], [50, 320]], dtype=np.float32)
    ordered = _order_corners(pts)
    sums = ordered.sum(axis=1)
    assert np.argmax(sums) == 2


def test_order_corners_returns_4_points():
    pts = np.array([[0, 0], [100, 0], [100, 80], [0, 80]], dtype=np.float32)
    ordered = _order_corners(pts)
    assert ordered.shape == (4, 2)


def test_correct_geometry_transform_composes_rotation_and_perspective():
    img = _image_with_horizontal_lines()
    rotated = _image_rotated(img, -5.0)
    result = correct_geometry(rotated)
    assert result.transform.shape == (3, 3)
