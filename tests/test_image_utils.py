import numpy as np
import pytest

from anonymizer.utils.image_utils import map_bbox_to_original, pad_bbox


IMG_300x200 = (200, 300, 3)


class TestPadBbox:
    def test_basic_padding_applied(self):
        result = pad_bbox((10, 10, 50, 50), IMG_300x200, padding=5)
        assert result == (5, 5, 55, 55)

    def test_top_left_clamped_to_zero(self):
        result = pad_bbox((2, 3, 50, 50), IMG_300x200, padding=10)
        assert result[0] == 0
        assert result[1] == 0

    def test_bottom_right_clamped_to_img_bounds(self):
        result = pad_bbox((10, 10, 295, 195), IMG_300x200, padding=10)
        assert result[2] == 299
        assert result[3] == 199

    def test_zero_padding_returns_same_values(self):
        bbox = (20, 30, 80, 90)
        assert pad_bbox(bbox, IMG_300x200, padding=0) == bbox

    def test_uses_default_blur_padding_px(self):
        from config import BLUR_PADDING_PX
        bbox = (20, 20, 80, 80)
        explicit = pad_bbox(bbox, IMG_300x200, padding=BLUR_PADDING_PX)
        default = pad_bbox(bbox, IMG_300x200)
        assert explicit == default

    def test_returns_tuple_of_four_ints(self):
        result = pad_bbox((10, 10, 50, 50), IMG_300x200, padding=5)
        assert isinstance(result, tuple)
        assert len(result) == 4
        assert all(isinstance(v, int) for v in result)

    def test_x1_always_lte_x2(self):
        result = pad_bbox((5, 5, 10, 10), IMG_300x200, padding=3)
        assert result[0] <= result[2]

    def test_y1_always_lte_y2(self):
        result = pad_bbox((5, 5, 10, 10), IMG_300x200, padding=3)
        assert result[1] <= result[3]

    def test_large_padding_clamped_all_sides(self):
        result = pad_bbox((50, 50, 100, 100), IMG_300x200, padding=1000)
        assert result == (0, 0, 299, 199)

    def test_width_and_height_used_correctly(self):
        shape = (100, 400, 3)
        result = pad_bbox((5, 5, 390, 90), shape, padding=20)
        assert result[2] == 399
        assert result[3] == 99


class TestMapBboxToOriginal:
    def _identity(self) -> np.ndarray:
        return np.eye(3, dtype=np.float64)

    def _translation(self, dx: float, dy: float) -> np.ndarray:
        T = np.eye(3, dtype=np.float64)
        T[0, 2] = dx
        T[1, 2] = dy
        return T

    def test_identity_transform_returns_same_bbox(self):
        bbox = (10, 20, 80, 90)
        result = map_bbox_to_original(bbox, self._identity(), (200, 300))
        assert result == bbox

    def test_translation_forward_inverted_correctly(self):
        dx, dy = 10, 20
        bbox_corrected = (30, 50, 90, 110)
        result = map_bbox_to_original(bbox_corrected, self._translation(dx, dy), (300, 400))
        assert result == (20, 30, 80, 90)

    def test_returns_tuple_of_four_ints(self):
        result = map_bbox_to_original((10, 10, 50, 50), self._identity(), (200, 300))
        assert isinstance(result, tuple)
        assert len(result) == 4
        assert all(isinstance(v, int) for v in result)

    def test_result_clamped_to_original_shape(self):
        T = self._translation(-50, -50)
        bbox = (0, 0, 50, 50)
        result = map_bbox_to_original(bbox, T, (100, 100))
        assert result[0] >= 0
        assert result[1] >= 0
        assert result[2] <= 99
        assert result[3] <= 99

    def test_x1_lte_x2_after_mapping(self):
        result = map_bbox_to_original((20, 20, 60, 60), self._identity(), (200, 300))
        assert result[0] <= result[2]

    def test_y1_lte_y2_after_mapping(self):
        result = map_bbox_to_original((20, 20, 60, 60), self._identity(), (200, 300))
        assert result[1] <= result[3]

    def test_rotation_180_maps_corners_correctly(self):
        h, w = 100, 200
        cx, cy = w / 2.0, h / 2.0
        angle_rad = np.pi
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        T = np.array([
            [cos_a, -sin_a, cx * (1 - cos_a) + cy * sin_a],
            [sin_a,  cos_a, cy * (1 - cos_a) - cx * sin_a],
            [0,      0,     1],
        ], dtype=np.float64)
        bbox = (10, 10, 30, 30)
        result = map_bbox_to_original(bbox, T, (h, w))
        assert result[0] >= 0 and result[1] >= 0
        assert result[2] <= w - 1 and result[3] <= h - 1

    def test_non_square_bbox_mapped(self):
        bbox = (0, 0, 200, 10)
        result = map_bbox_to_original(bbox, self._identity(), (200, 300))
        assert result == (0, 0, 200, 10)

    def test_original_shape_respects_height_width_order(self):
        result = map_bbox_to_original((0, 0, 5, 5), self._identity(), (50, 100))
        assert result[2] <= 99
        assert result[3] <= 49
