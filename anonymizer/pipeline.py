from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

from anonymizer.ocr.engine import run_ocr
from anonymizer.pii.detector import OcrToken, detect
from anonymizer.preprocessing.enhance import enhance
from anonymizer.preprocessing.geometry import correct_geometry
from anonymizer.preprocessing.loader import load
from anonymizer.preprocessing.upscale import upscale
from anonymizer.redaction.blur import redact

_UPSCALE_FACTOR = 4


@dataclass(frozen=True)
class PipelineResult:
    output_path: Path
    tokens_found: int
    regions_redacted: int


def run(input_path: str | Path, output_path: str | Path) -> PipelineResult:
    output_path = Path(output_path)

    load_result = load(input_path)
    original = load_result.image

    enhanced = enhance(original)

    geo_result = correct_geometry(enhanced)
    corrected = geo_result.image
    transform = geo_result.transform

    upscale_result = upscale(corrected)
    ocr_input = upscale_result.image

    raw_tokens = run_ocr(ocr_input)
    logger.debug("OCR found %d token(s)", len(raw_tokens))
    for tok in raw_tokens:
        logger.debug("  token: %r  bbox=%s", tok.text, tok.bbox)

    if upscale_result.was_upscaled:
        tokens = _scale_tokens(raw_tokens, 1.0 / _UPSCALE_FACTOR)
    else:
        tokens = raw_tokens

    detection = detect(tokens)

    redact_result = redact(original, detection.redact_boxes, transform)

    _save(redact_result.image, output_path)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return PipelineResult(
        output_path=output_path,
        tokens_found=len(tokens),
        regions_redacted=len(detection.redact_boxes),
    )


def _scale_tokens(tokens: list[OcrToken], factor: float) -> list[OcrToken]:
    return [
        OcrToken(
            text=tok.text,
            bbox=(
                int(tok.bbox[0] * factor),
                int(tok.bbox[1] * factor),
                int(tok.bbox[2] * factor),
                int(tok.bbox[3] * factor),
            ),
        )
        for tok in tokens
    ]


def _save(img: np.ndarray, path: Path) -> None:
    ext = path.suffix.lower() if path.suffix else ".jpg"
    success, buf = cv2.imencode(ext, img)
    if not success:
        raise ValueError(f"Failed to encode image as {ext}")
    path.write_bytes(buf.tobytes())
