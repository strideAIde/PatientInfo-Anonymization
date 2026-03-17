from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from anonymizer.pipeline import PipelineResult, _scale_tokens, run
from anonymizer.pii.detector import OcrToken


def _write_jpg(path: Path, h: int = 200, w: int = 300, val: int = 128) -> Path:
    img = np.full((h, w, 3), val, dtype=np.uint8)
    img[::2, ::2] = 0
    cv2.imwrite(str(path), img)
    return path


def _identity() -> np.ndarray:
    return np.eye(3, dtype=np.float64)


class TestPipelineResult:
    def test_returns_pipeline_result(self, tmp_path):
        src = _write_jpg(tmp_path / "in.jpg")
        dst = tmp_path / "out.jpg"
        with patch("anonymizer.pipeline.run_ocr", return_value=[]):
            result = run(src, dst)
        assert isinstance(result, PipelineResult)

    def test_output_path_stored(self, tmp_path):
        src = _write_jpg(tmp_path / "in.jpg")
        dst = tmp_path / "out.jpg"
        with patch("anonymizer.pipeline.run_ocr", return_value=[]):
            result = run(src, dst)
        assert result.output_path == dst

    def test_accepts_string_paths(self, tmp_path):
        src = _write_jpg(tmp_path / "in.jpg")
        dst = tmp_path / "out.jpg"
        with patch("anonymizer.pipeline.run_ocr", return_value=[]):
            result = run(str(src), str(dst))
        assert isinstance(result, PipelineResult)


class TestOutputFile:
    def test_output_file_created(self, tmp_path):
        src = _write_jpg(tmp_path / "in.jpg")
        dst = tmp_path / "out.jpg"
        with patch("anonymizer.pipeline.run_ocr", return_value=[]):
            run(src, dst)
        assert dst.exists()

    def test_output_file_is_valid_image(self, tmp_path):
        src = _write_jpg(tmp_path / "in.jpg")
        dst = tmp_path / "out.jpg"
        with patch("anonymizer.pipeline.run_ocr", return_value=[]):
            run(src, dst)
        img = cv2.imread(str(dst))
        assert img is not None
        assert img.dtype == np.uint8

    def test_output_same_shape_as_input(self, tmp_path):
        src = _write_jpg(tmp_path / "in.jpg", h=120, w=160)
        dst = tmp_path / "out.jpg"
        with patch("anonymizer.pipeline.run_ocr", return_value=[]):
            run(src, dst)
        out = cv2.imread(str(dst))
        assert out.shape[:2] == (120, 160)


class TestNoPii:
    def test_no_tokens_zero_regions_redacted(self, tmp_path):
        src = _write_jpg(tmp_path / "in.jpg")
        dst = tmp_path / "out.jpg"
        with patch("anonymizer.pipeline.run_ocr", return_value=[]):
            result = run(src, dst)
        assert result.regions_redacted == 0

    def test_no_tokens_zero_tokens_found(self, tmp_path):
        src = _write_jpg(tmp_path / "in.jpg")
        dst = tmp_path / "out.jpg"
        with patch("anonymizer.pipeline.run_ocr", return_value=[]):
            result = run(src, dst)
        assert result.tokens_found == 0

    def test_no_pii_labels_no_redaction(self, tmp_path):
        src = _write_jpg(tmp_path / "in.jpg")
        dst = tmp_path / "out.jpg"
        non_pii = [OcrToken("Hello", (0, 0, 50, 20)), OcrToken("World", (60, 0, 120, 20))]
        with patch("anonymizer.pipeline.run_ocr", return_value=non_pii):
            result = run(src, dst)
        assert result.regions_redacted == 0
        assert result.tokens_found == 2


class TestPiiDetected:
    def test_name_label_value_counted(self, tmp_path):
        src = _write_jpg(tmp_path / "in.jpg", h=1200, w=1600)
        dst = tmp_path / "out.jpg"
        tokens = [
            OcrToken("Name", (0, 0, 50, 20)),
            OcrToken("John Smith", (60, 0, 180, 20)),
        ]
        with patch("anonymizer.pipeline.run_ocr", return_value=tokens):
            result = run(src, dst)
        assert result.regions_redacted == 1

    def test_tokens_found_reflects_ocr_count(self, tmp_path):
        src = _write_jpg(tmp_path / "in.jpg", h=1200, w=1600)
        dst = tmp_path / "out.jpg"
        tokens = [
            OcrToken("Name", (0, 0, 50, 20)),
            OcrToken("John", (60, 0, 110, 20)),
            OcrToken("Smith", (120, 0, 180, 20)),
        ]
        with patch("anonymizer.pipeline.run_ocr", return_value=tokens):
            result = run(src, dst)
        assert result.tokens_found == 3

    def test_pii_region_blurred_in_output(self, tmp_path):
        src = _write_jpg(tmp_path / "in.jpg", h=1200, w=1600)
        dst = tmp_path / "out.jpg"
        tokens = [
            OcrToken("Name", (0, 80, 50, 100)),
            OcrToken("John Smith", (60, 80, 200, 100)),
        ]
        original = cv2.imread(str(src))
        with patch("anonymizer.pipeline.run_ocr", return_value=tokens):
            run(src, dst)
        output = cv2.imread(str(dst))
        orig_roi = original[80:101, 60:201]
        out_roi = output[80:101, 60:201]
        assert not np.array_equal(orig_roi, out_roi)

    def test_multiple_pii_fields_all_counted(self, tmp_path):
        src = _write_jpg(tmp_path / "in.jpg", h=1200, w=1600)
        dst = tmp_path / "out.jpg"
        tokens = [
            OcrToken("Name", (0, 0, 50, 20)),
            OcrToken("John Smith", (60, 0, 180, 20)),
            OcrToken("PATIENT ID:", (0, 30, 100, 50)),
            OcrToken("KIER1234", (110, 30, 200, 50)),
        ]
        with patch("anonymizer.pipeline.run_ocr", return_value=tokens):
            result = run(src, dst)
        assert result.regions_redacted == 2


class TestErrorHandling:
    def test_missing_input_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run(tmp_path / "nonexistent.jpg", tmp_path / "out.jpg")


class TestScaleTokens:
    def test_scale_down_by_half(self):
        tokens = [OcrToken("A", (100, 200, 300, 400))]
        scaled = _scale_tokens(tokens, 0.5)
        assert scaled[0].bbox == (50, 100, 150, 200)

    def test_scale_by_quarter(self):
        tokens = [OcrToken("B", (40, 80, 120, 160))]
        scaled = _scale_tokens(tokens, 0.25)
        assert scaled[0].bbox == (10, 20, 30, 40)

    def test_text_preserved(self):
        tokens = [OcrToken("Name", (0, 0, 50, 20))]
        scaled = _scale_tokens(tokens, 0.5)
        assert scaled[0].text == "Name"

    def test_empty_list(self):
        assert _scale_tokens([], 0.5) == []

    def test_multiple_tokens(self):
        tokens = [OcrToken("A", (0, 0, 40, 20)), OcrToken("B", (50, 0, 100, 20))]
        scaled = _scale_tokens(tokens, 0.5)
        assert len(scaled) == 2

    def test_scale_one_is_identity(self):
        tokens = [OcrToken("A", (10, 20, 50, 80))]
        scaled = _scale_tokens(tokens, 1.0)
        assert scaled[0].bbox == (10, 20, 50, 80)
