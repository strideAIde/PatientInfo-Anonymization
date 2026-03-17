from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import anonymizer.ocr.engine as engine_module
from anonymizer.ocr.engine import reset_model_cache, run_ocr, _infer
from anonymizer.pii.detector import OcrToken

IMG_H, IMG_W = 200, 300


def setup_function():
    reset_model_cache()


def _img(h: int = IMG_H, w: int = IMG_W) -> np.ndarray:
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _quad(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


class TestRunOcr:
    def test_returns_list(self):
        with patch.object(engine_module, "_get_reader", return_value=None):
            result = run_ocr(_img())
        assert isinstance(result, list)

    def test_reader_unavailable_returns_empty(self):
        with patch.object(engine_module, "_get_reader", return_value=None):
            assert run_ocr(_img()) == []

    def test_delegates_to_infer(self):
        fake_tokens = [OcrToken(text="Name", bbox=(0, 0, 50, 20))]
        mock_reader = MagicMock()
        with patch.object(engine_module, "_get_reader", return_value=mock_reader), \
             patch.object(engine_module, "_infer", return_value=fake_tokens):
            result = run_ocr(_img())
        assert result == fake_tokens

    def test_returns_ocr_token_instances(self):
        fake_tokens = [OcrToken(text="ID", bbox=(10, 10, 80, 30))]
        mock_reader = MagicMock()
        with patch.object(engine_module, "_get_reader", return_value=mock_reader), \
             patch.object(engine_module, "_infer", return_value=fake_tokens):
            result = run_ocr(_img())
        assert all(isinstance(t, OcrToken) for t in result)


class TestInfer:
    def _run(self, easyocr_results, h=IMG_H, w=IMG_W):
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = easyocr_results
        img = np.zeros((h, w, 3), dtype=np.uint8)
        return _infer(mock_reader, img)

    def test_valid_single_result(self):
        results = [(_quad(10, 20, 100, 40), "Name", 0.9)]
        tokens = self._run(results)
        assert len(tokens) == 1
        assert tokens[0].text == "Name"
        assert tokens[0].bbox == (10, 20, 100, 40)

    def test_valid_multiple_results(self):
        results = [
            (_quad(0, 0, 50, 20), "Name", 0.95),
            (_quad(60, 0, 130, 20), "KIER123", 0.8),
        ]
        tokens = self._run(results)
        assert len(tokens) == 2

    def test_text_stripped(self):
        results = [(_quad(0, 0, 50, 20), "  Name  ", 0.9)]
        tokens = self._run(results)
        assert tokens[0].text == "Name"

    def test_score_below_threshold_filtered(self):
        results = [(_quad(0, 0, 50, 20), "Name", 0.1)]
        assert self._run(results) == []

    def test_score_at_threshold_included(self):
        from config import OCR_CONFIDENCE_THRESHOLD
        results = [(_quad(0, 0, 50, 20), "Name", OCR_CONFIDENCE_THRESHOLD)]
        assert len(self._run(results)) == 1

    def test_empty_text_skipped(self):
        results = [(_quad(0, 0, 50, 20), "", 0.9)]
        assert self._run(results) == []

    def test_whitespace_only_text_skipped(self):
        results = [(_quad(0, 0, 50, 20), "   ", 0.9)]
        assert self._run(results) == []

    def test_degenerate_bbox_x1_eq_x2_skipped(self):
        results = [(_quad(50, 0, 50, 20), "Name", 0.9)]
        assert self._run(results) == []

    def test_degenerate_bbox_y1_eq_y2_skipped(self):
        results = [(_quad(0, 20, 50, 20), "Name", 0.9)]
        assert self._run(results) == []

    def test_bbox_clamped_to_image_width(self):
        results = [(_quad(0, 0, 500, 20), "Name", 0.9)]
        tokens = self._run(results)
        assert tokens[0].bbox[2] <= IMG_W - 1

    def test_bbox_clamped_to_image_height(self):
        results = [(_quad(0, 0, 50, 500), "Name", 0.9)]
        tokens = self._run(results)
        assert tokens[0].bbox[3] <= IMG_H - 1

    def test_negative_coords_clamped_to_zero(self):
        results = [(_quad(-10, -5, 50, 20), "Name", 0.9)]
        tokens = self._run(results)
        assert tokens[0].bbox[0] == 0
        assert tokens[0].bbox[1] == 0

    def test_quad_min_max_extraction(self):
        skewed_quad = [[5, 10], [100, 8], [102, 42], [3, 44]]
        mock_reader = MagicMock()
        mock_reader.readtext.return_value = [(skewed_quad, "Text", 0.9)]
        img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        tokens = _infer(mock_reader, img)
        assert tokens[0].bbox == (3, 8, 102, 44)

    def test_empty_results_returns_empty(self):
        assert self._run([]) == []

    def test_returns_ocr_token_instances(self):
        results = [(_quad(0, 0, 50, 20), "Name", 0.9)]
        tokens = self._run(results)
        assert all(isinstance(t, OcrToken) for t in tokens)


class TestResetModelCache:
    def test_reset_clears_reader(self):
        engine_module._reader = MagicMock()
        engine_module._reader_loaded = True
        reset_model_cache()
        assert engine_module._reader is None

    def test_reset_clears_loaded_flag(self):
        engine_module._reader_loaded = True
        reset_model_cache()
        assert engine_module._reader_loaded is False

    def test_reset_allows_reload(self):
        engine_module._reader_loaded = True
        reset_model_cache()
        assert not engine_module._reader_loaded


class TestSingleton:
    def test_get_reader_called_each_run_ocr(self):
        call_count = 0

        def fake_get_reader():
            nonlocal call_count
            call_count += 1
            return None

        with patch.object(engine_module, "_get_reader", side_effect=fake_get_reader):
            run_ocr(_img())
            run_ocr(_img())

        assert call_count == 2

    def test_reader_loaded_flag_prevents_reload(self):
        engine_module._reader_loaded = True
        engine_module._reader = None
        reader = engine_module._get_reader()
        assert reader is None

    def test_reader_loaded_flag_set_after_init(self):
        engine_module._reader_loaded = False
        with patch("easyocr.Reader", return_value=MagicMock()):
            engine_module._get_reader()
        assert engine_module._reader_loaded is True
