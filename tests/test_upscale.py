from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import anonymizer.preprocessing.upscale as upscale_module
from anonymizer.preprocessing.upscale import UpscaleResult, upscale, _run_esrgan


def _img(h: int = 200, w: int = 300, val: int = 128) -> np.ndarray:
    return np.full((h, w, 3), val, dtype=np.uint8)


def setup_function():
    upscale_module.reset_model_cache()


def test_upscale_returns_upscale_result():
    with patch.object(upscale_module, "UPSCALE_ENABLED", False):
        result = upscale(_img())
    assert isinstance(result, UpscaleResult)


def test_disabled_was_upscaled_false():
    with patch.object(upscale_module, "UPSCALE_ENABLED", False):
        result = upscale(_img())
    assert result.was_upscaled is False


def test_disabled_returns_copy_not_same_object():
    img = _img()
    with patch.object(upscale_module, "UPSCALE_ENABLED", False):
        result = upscale(img)
    assert result.image is not img


def test_disabled_pixels_unchanged():
    img = _img(val=64)
    with patch.object(upscale_module, "UPSCALE_ENABLED", False):
        result = upscale(img)
    np.testing.assert_array_equal(result.image, img)


def test_disabled_does_not_mutate_input():
    img = _img()
    original = img.copy()
    with patch.object(upscale_module, "UPSCALE_ENABLED", False):
        upscale(img)
    np.testing.assert_array_equal(img, original)


def test_large_image_skipped():
    img = _img(h=300, w=400)
    with patch.object(upscale_module, "UPSCALE_THRESHOLD_PX", 200):
        result = upscale(img)
    assert result.was_upscaled is False


def test_shortest_edge_at_threshold_skipped():
    img = _img(h=200, w=400)
    with patch.object(upscale_module, "UPSCALE_THRESHOLD_PX", 200):
        result = upscale(img)
    assert result.was_upscaled is False


def test_model_unavailable_falls_back():
    img = _img(h=100, w=150)
    with patch.object(upscale_module, "UPSCALE_THRESHOLD_PX", 200), \
         patch.object(upscale_module, "_get_model", return_value=None):
        result = upscale(img)
    assert result.was_upscaled is False
    assert result.image.shape == img.shape


def test_model_unavailable_returns_copy_not_same_object():
    img = _img(h=100, w=150)
    with patch.object(upscale_module, "UPSCALE_THRESHOLD_PX", 200), \
         patch.object(upscale_module, "_get_model", return_value=None):
        result = upscale(img)
    assert result.image is not img


def test_small_image_upscaled_with_mock():
    img = _img(h=100, w=150)
    fake_output = _img(h=400, w=600)
    mock_model = MagicMock()
    with patch.object(upscale_module, "UPSCALE_THRESHOLD_PX", 200), \
         patch.object(upscale_module, "_get_model", return_value=mock_model), \
         patch.object(upscale_module, "_run_esrgan", return_value=fake_output):
        result = upscale(img)
    assert result.was_upscaled is True
    assert result.image.shape == fake_output.shape


def test_upscaled_output_dtype_uint8():
    with patch.object(upscale_module, "UPSCALE_ENABLED", False):
        result = upscale(_img())
    assert result.image.dtype == np.uint8


def test_run_esrgan_converts_bgr_to_rgb_and_back():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, :, 0] = 100

    captured = {}

    def fake_enhance(rgb_img, outscale):
        captured["input"] = rgb_img.copy()
        return rgb_img, None

    mock_model = MagicMock()
    mock_model.enhance.side_effect = fake_enhance

    result = _run_esrgan(mock_model, img)

    assert captured["input"][:, :, 2].mean() == pytest.approx(100, abs=1)
    assert result.dtype == np.uint8


def test_reset_model_cache_clears_state():
    upscale_module._model = MagicMock()
    upscale_module._model_loaded = True
    upscale_module.reset_model_cache()
    assert upscale_module._model is None
    assert upscale_module._model_loaded is False
