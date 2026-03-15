import numpy as np
import pytest

from anonymizer.redaction.blur import RedactResult, _apply_blur, redact


def _identity() -> np.ndarray:
    return np.eye(3, dtype=np.float64)


def _checkerboard(h: int, w: int) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[::2, ::2] = 255
    img[1::2, 1::2] = 255
    return img


def _solid(h: int, w: int, val: int = 128) -> np.ndarray:
    return np.full((h, w, 3), val, dtype=np.uint8)


class TestRedactReturnType:
    def test_returns_redact_result(self):
        assert isinstance(redact(_solid(100, 100), [], _identity()), RedactResult)

    def test_image_field_is_ndarray(self):
        result = redact(_solid(100, 100), [], _identity())
        assert isinstance(result.image, np.ndarray)

    def test_output_dtype_uint8(self):
        result = redact(_solid(100, 100), [], _identity())
        assert result.image.dtype == np.uint8

    def test_output_shape_matches_input(self):
        img = _solid(80, 120)
        result = redact(img, [], _identity())
        assert result.image.shape == img.shape


class TestNoMutation:
    def test_input_not_mutated_empty_boxes(self):
        img = _checkerboard(100, 100)
        original = img.copy()
        redact(img, [], _identity())
        np.testing.assert_array_equal(img, original)

    def test_input_not_mutated_with_boxes(self):
        img = _checkerboard(100, 100)
        original = img.copy()
        redact(img, [(10, 10, 50, 50)], _identity())
        np.testing.assert_array_equal(img, original)

    def test_output_is_copy_not_same_object(self):
        img = _solid(100, 100)
        result = redact(img, [], _identity())
        assert result.image is not img


class TestEmptyBoxes:
    def test_empty_boxes_output_equals_input(self):
        img = _checkerboard(100, 100)
        result = redact(img, [], _identity())
        np.testing.assert_array_equal(result.image, img)


class TestBlurApplied:
    def test_blurred_region_differs_from_original(self):
        img = _checkerboard(200, 200)
        result = redact(img, [(20, 20, 80, 80)], _identity())
        roi_original = img[20:81, 20:81]
        roi_blurred = result.image[20:81, 20:81]
        assert not np.array_equal(roi_original, roi_blurred)

    def test_pixels_outside_bbox_unchanged(self):
        img = _checkerboard(200, 200)
        result = redact(img, [(50, 50, 100, 100)], _identity())
        np.testing.assert_array_equal(result.image[:40, :40], img[:40, :40])
        np.testing.assert_array_equal(result.image[120:, 120:], img[120:, 120:])

    def test_blurred_values_intermediate(self):
        img = _checkerboard(100, 100)
        result = redact(img, [(10, 10, 60, 60)], _identity())
        roi = result.image[10:61, 10:61].astype(np.float32)
        assert roi.min() > 0
        assert roi.max() < 255

    def test_multiple_boxes_all_blurred(self):
        img = _checkerboard(200, 200)
        boxes = [(10, 10, 40, 40), (100, 100, 150, 150)]
        result = redact(img, boxes, _identity())
        for x1, y1, x2, y2 in boxes:
            roi_orig = img[y1 : y2 + 1, x1 : x2 + 1]
            roi_blur = result.image[y1 : y2 + 1, x1 : x2 + 1]
            assert not np.array_equal(roi_orig, roi_blur)


class TestTransformMapping:
    def _translation(self, dx: float, dy: float) -> np.ndarray:
        T = np.eye(3, dtype=np.float64)
        T[0, 2] = dx
        T[1, 2] = dy
        return T

    def test_identity_transform_blurs_same_region(self):
        img = _checkerboard(200, 200)
        box = (20, 20, 80, 80)
        result = redact(img, [box], _identity())
        roi_orig = img[20:81, 20:81]
        roi_blur = result.image[20:81, 20:81]
        assert not np.array_equal(roi_orig, roi_blur)

    def test_translation_shifts_blurred_region(self):
        img = _checkerboard(300, 300)
        dx, dy = 30, 30
        T = self._translation(dx, dy)
        box_corrected = (60, 60, 120, 120)
        result = redact(img, [box_corrected], T)
        expected_x1, expected_y1 = 60 - dx, 60 - dy
        expected_x2, expected_y2 = 120 - dx, 120 - dy
        roi_orig = img[expected_y1 : expected_y2 + 1, expected_x1 : expected_x2 + 1]
        roi_blur = result.image[expected_y1 : expected_y2 + 1, expected_x1 : expected_x2 + 1]
        assert not np.array_equal(roi_orig, roi_blur)

    def test_unrelated_region_not_blurred_under_translation(self):
        img = _checkerboard(300, 300)
        T = self._translation(50, 50)
        box_corrected = (100, 100, 150, 150)
        result = redact(img, [box_corrected], T)
        np.testing.assert_array_equal(result.image[200:250, 200:250], img[200:250, 200:250])


class TestApplyBlurInternal:
    def test_apply_blur_modifies_in_place(self):
        img = _checkerboard(100, 100)
        original_roi = img[10:51, 10:51].copy()
        _apply_blur(img, (10, 10, 50, 50))
        assert not np.array_equal(img[10:51, 10:51], original_roi)

    def test_apply_blur_leaves_exterior_unchanged(self):
        img = _solid(100, 100, val=200)
        corner = img[:5, :5].copy()
        _apply_blur(img, (20, 20, 80, 80))
        np.testing.assert_array_equal(img[:5, :5], corner)
