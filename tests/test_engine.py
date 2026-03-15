from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import anonymizer.ocr.engine as engine_module
from anonymizer.ocr.engine import _parse_output, reset_model_cache, run_ocr
from anonymizer.pii.detector import OcrToken

IMG_SHAPE = (200, 300, 3)


def setup_function():
    reset_model_cache()


def _img(h: int = 100, w: int = 150) -> np.ndarray:
    return np.full((h, w, 3), 128, dtype=np.uint8)


class TestRunOcr:
    def test_returns_list(self):
        with patch.object(engine_module, "_get_model", return_value=(None, None)):
            result = run_ocr(_img())
        assert isinstance(result, list)

    def test_model_unavailable_returns_empty(self):
        with patch.object(engine_module, "_get_model", return_value=(None, None)):
            assert run_ocr(_img()) == []

    def test_processor_none_returns_empty(self):
        mock_model = MagicMock()
        with patch.object(engine_module, "_get_model", return_value=(None, mock_model)):
            assert run_ocr(_img()) == []

    def test_delegates_to_infer(self):
        fake_tokens = [OcrToken(text="Name", bbox=(0, 0, 50, 20))]
        mock_proc = MagicMock()
        mock_model = MagicMock()
        with patch.object(engine_module, "_get_model", return_value=(mock_proc, mock_model)), \
             patch.object(engine_module, "_infer", return_value=fake_tokens):
            result = run_ocr(_img())
        assert result == fake_tokens

    def test_returns_ocr_token_instances(self):
        fake_tokens = [OcrToken(text="ID", bbox=(10, 10, 80, 30))]
        mock_proc = MagicMock()
        mock_model = MagicMock()
        with patch.object(engine_module, "_get_model", return_value=(mock_proc, mock_model)), \
             patch.object(engine_module, "_infer", return_value=fake_tokens):
            result = run_ocr(_img())
        assert all(isinstance(t, OcrToken) for t in result)


class TestParseOutput:
    def test_valid_single_record(self):
        raw = '[{"text": "Name", "bbox": [10, 20, 100, 40], "score": 0.9}]'
        tokens = _parse_output(raw, IMG_SHAPE)
        assert len(tokens) == 1
        assert tokens[0].text == "Name"
        assert tokens[0].bbox == (10, 20, 100, 40)

    def test_valid_multiple_records(self):
        raw = (
            '[{"text": "Name", "bbox": [0, 0, 50, 20], "score": 0.95},'
            ' {"text": "KIER123", "bbox": [60, 0, 130, 20], "score": 0.8}]'
        )
        tokens = _parse_output(raw, IMG_SHAPE)
        assert len(tokens) == 2

    def test_text_field_stripped(self):
        raw = '[{"text": "  Name  ", "bbox": [0, 0, 50, 20], "score": 0.9}]'
        tokens = _parse_output(raw, IMG_SHAPE)
        assert tokens[0].text == "Name"

    def test_score_below_threshold_filtered(self):
        raw = '[{"text": "Name", "bbox": [0, 0, 50, 20], "score": 0.3}]'
        tokens = _parse_output(raw, IMG_SHAPE)
        assert tokens == []

    def test_score_at_threshold_included(self):
        from config import OCR_CONFIDENCE_THRESHOLD
        raw = f'[{{"text": "Name", "bbox": [0, 0, 50, 20], "score": {OCR_CONFIDENCE_THRESHOLD}}}]'
        tokens = _parse_output(raw, IMG_SHAPE)
        assert len(tokens) == 1

    def test_missing_score_defaults_to_included(self):
        raw = '[{"text": "Name", "bbox": [0, 0, 50, 20]}]'
        tokens = _parse_output(raw, IMG_SHAPE)
        assert len(tokens) == 1

    def test_empty_text_skipped(self):
        raw = '[{"text": "", "bbox": [0, 0, 50, 20], "score": 0.9}]'
        assert _parse_output(raw, IMG_SHAPE) == []

    def test_whitespace_only_text_skipped(self):
        raw = '[{"text": "   ", "bbox": [0, 0, 50, 20], "score": 0.9}]'
        assert _parse_output(raw, IMG_SHAPE) == []

    def test_bbox_wrong_length_skipped(self):
        raw = '[{"text": "Name", "bbox": [0, 0, 50], "score": 0.9}]'
        assert _parse_output(raw, IMG_SHAPE) == []

    def test_bbox_missing_skipped(self):
        raw = '[{"text": "Name", "score": 0.9}]'
        assert _parse_output(raw, IMG_SHAPE) == []

    def test_degenerate_bbox_x1_eq_x2_skipped(self):
        raw = '[{"text": "Name", "bbox": [50, 0, 50, 20], "score": 0.9}]'
        assert _parse_output(raw, IMG_SHAPE) == []

    def test_degenerate_bbox_y1_eq_y2_skipped(self):
        raw = '[{"text": "Name", "bbox": [0, 20, 50, 20], "score": 0.9}]'
        assert _parse_output(raw, IMG_SHAPE) == []

    def test_bbox_clamped_to_image_width(self):
        raw = '[{"text": "Name", "bbox": [0, 0, 500, 20], "score": 0.9}]'
        tokens = _parse_output(raw, IMG_SHAPE)
        assert tokens[0].bbox[2] <= IMG_SHAPE[1] - 1

    def test_bbox_clamped_to_image_height(self):
        raw = '[{"text": "Name", "bbox": [0, 0, 50, 500], "score": 0.9}]'
        tokens = _parse_output(raw, IMG_SHAPE)
        assert tokens[0].bbox[3] <= IMG_SHAPE[0] - 1

    def test_negative_coords_clamped_to_zero(self):
        raw = '[{"text": "Name", "bbox": [-10, -5, 50, 20], "score": 0.9}]'
        tokens = _parse_output(raw, IMG_SHAPE)
        assert tokens[0].bbox[0] == 0
        assert tokens[0].bbox[1] == 0

    def test_no_json_array_returns_empty(self):
        assert _parse_output("no JSON here", IMG_SHAPE) == []

    def test_malformed_json_returns_empty(self):
        assert _parse_output("[{bad json}]", IMG_SHAPE) == []

    def test_empty_string_returns_empty(self):
        assert _parse_output("", IMG_SHAPE) == []

    def test_empty_array_returns_empty(self):
        assert _parse_output("[]", IMG_SHAPE) == []

    def test_non_dict_record_skipped(self):
        raw = '["not_a_dict", {"text": "Name", "bbox": [0, 0, 50, 20], "score": 0.9}]'
        tokens = _parse_output(raw, IMG_SHAPE)
        assert len(tokens) == 1

    def test_json_embedded_in_surrounding_text(self):
        raw = 'Here is the result: [{"text": "ID", "bbox": [0, 0, 30, 20], "score": 0.8}] done.'
        tokens = _parse_output(raw, IMG_SHAPE)
        assert len(tokens) == 1
        assert tokens[0].text == "ID"

    def test_returns_ocr_token_instances(self):
        raw = '[{"text": "Name", "bbox": [0, 0, 50, 20], "score": 0.9}]'
        tokens = _parse_output(raw, IMG_SHAPE)
        assert all(isinstance(t, OcrToken) for t in tokens)


class TestResetModelCache:
    def test_reset_clears_processor(self):
        engine_module._processor = MagicMock()
        engine_module._model_loaded = True
        reset_model_cache()
        assert engine_module._processor is None

    def test_reset_clears_model(self):
        engine_module._model = MagicMock()
        engine_module._model_loaded = True
        reset_model_cache()
        assert engine_module._model is None

    def test_reset_clears_loaded_flag(self):
        engine_module._model_loaded = True
        reset_model_cache()
        assert engine_module._model_loaded is False


class TestSingleton:
    def test_get_model_not_called_twice(self):
        call_count = 0

        def fake_get_model():
            nonlocal call_count
            call_count += 1
            return None, None

        with patch.object(engine_module, "_get_model", side_effect=fake_get_model):
            run_ocr(_img())
            run_ocr(_img())

        assert call_count == 2

    def test_model_loaded_flag_prevents_reload(self):
        engine_module._model_loaded = True
        engine_module._processor = None
        engine_module._model = None
        proc, model = engine_module._get_model()
        assert proc is None
        assert model is None
