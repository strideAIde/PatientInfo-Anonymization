from __future__ import annotations

import json
import logging

import cv2
import numpy as np
from PIL import Image

from config import OCR_CONFIDENCE_THRESHOLD, OCR_MODEL_ID
from anonymizer.pii.detector import OcrToken

logger = logging.getLogger(__name__)

_SPOTTING_PROMPT = (
    "Perform OCR text spotting on this image. "
    "Return a JSON array where every element has three fields: "
    "'text' (the recognised string), "
    "'bbox' ([x1, y1, x2, y2] integer pixel coordinates), "
    "and 'score' (confidence between 0 and 1). "
    "Output only the JSON array, nothing else."
)

_processor = None
_model = None
_model_loaded = False


def run_ocr(img: np.ndarray) -> list[OcrToken]:
    processor, model = _get_model()
    if processor is None or model is None:
        return []
    return _infer(processor, model, img)


def _get_model():
    global _processor, _model, _model_loaded
    if _model_loaded:
        return _processor, _model
    _model_loaded = True
    try:
        from transformers import AutoModelForVision2Seq, AutoProcessor

        _processor = AutoProcessor.from_pretrained(OCR_MODEL_ID, trust_remote_code=True)
        _model = AutoModelForVision2Seq.from_pretrained(
            OCR_MODEL_ID, trust_remote_code=True
        )
        _model.eval()
    except Exception as e:
        logger.warning("OCR model unavailable: %s", e)
        _processor = None
        _model = None
    return _processor, _model


def _infer(processor, model, img: np.ndarray) -> list[OcrToken]:
    import torch

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    inputs = processor(text=_SPOTTING_PROMPT, images=pil_img, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=2048)
    raw = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return _parse_output(raw, img.shape)


def _parse_output(raw: str, img_shape: tuple) -> list[OcrToken]:
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start == -1 or end == 0:
        return []
    try:
        records = json.loads(raw[start:end])
    except json.JSONDecodeError:
        logger.warning("OCR output is not valid JSON")
        return []
    if not isinstance(records, list):
        return []

    h, w = img_shape[:2]
    tokens: list[OcrToken] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        text = str(rec.get("text", "")).strip()
        if not text:
            continue
        score = float(rec.get("score", 1.0))
        if score < OCR_CONFIDENCE_THRESHOLD:
            continue
        bbox = rec.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = (int(v) for v in bbox)
        except (TypeError, ValueError):
            continue
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        if x1 >= x2 or y1 >= y2:
            continue
        tokens.append(OcrToken(text=text, bbox=(x1, y1, x2, y2)))
    return tokens


def reset_model_cache() -> None:
    global _processor, _model, _model_loaded
    _processor = None
    _model = None
    _model_loaded = False
