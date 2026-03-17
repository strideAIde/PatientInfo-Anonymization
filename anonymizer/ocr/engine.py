from __future__ import annotations

import logging
import warnings

import numpy as np

from config import OCR_CONFIDENCE_THRESHOLD, OCR_LANGUAGES, OCR_USE_GPU
from anonymizer.pii.detector import OcrToken

logger = logging.getLogger(__name__)

_reader = None
_reader_loaded = False


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _resolve_gpu() -> bool:
    if OCR_USE_GPU is not None:
        return OCR_USE_GPU
    return _cuda_available()


def run_ocr(img: np.ndarray) -> list[OcrToken]:
    reader = _get_reader()
    if reader is None:
        return []
    return _infer(reader, img)


def _get_reader():
    global _reader, _reader_loaded
    if _reader_loaded:
        return _reader
    _reader_loaded = True
    use_gpu = _resolve_gpu()
    logger.info("EasyOCR initializing (gpu=%s)", use_gpu)
    try:
        import easyocr
        if not use_gpu:
            warnings.filterwarnings(
                "ignore",
                message=".*pin_memory.*no accelerator.*",
                category=UserWarning,
            )
        _reader = easyocr.Reader(OCR_LANGUAGES, gpu=use_gpu)
    except Exception as e:
        logger.warning("EasyOCR unavailable: %s", e)
        _reader = None
    return _reader


def _infer(reader, img: np.ndarray) -> list[OcrToken]:
    results = reader.readtext(img)
    tokens: list[OcrToken] = []
    h, w = img.shape[:2]
    for (quad, text, score) in results:
        if score < OCR_CONFIDENCE_THRESHOLD:
            continue
        text = text.strip()
        if not text:
            continue
        xs = [pt[0] for pt in quad]
        ys = [pt[1] for pt in quad]
        x1 = max(0, min(int(min(xs)), w - 1))
        y1 = max(0, min(int(min(ys)), h - 1))
        x2 = max(0, min(int(max(xs)), w - 1))
        y2 = max(0, min(int(max(ys)), h - 1))
        if x1 >= x2 or y1 >= y2:
            continue
        tokens.append(OcrToken(text=text, bbox=(x1, y1, x2, y2)))
    return tokens


def reset_model_cache() -> None:
    global _reader, _reader_loaded
    _reader = None
    _reader_loaded = False
