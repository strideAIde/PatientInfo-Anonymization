from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config import UPSCALE_ENABLED, UPSCALE_THRESHOLD_PX

logger = logging.getLogger(__name__)

_model = None
_model_loaded = False


@dataclass(frozen=True)
class UpscaleResult:
    image: np.ndarray
    was_upscaled: bool


def upscale(img: np.ndarray) -> UpscaleResult:
    if not UPSCALE_ENABLED:
        return UpscaleResult(image=img.copy(), was_upscaled=False)

    h, w = img.shape[:2]
    if min(h, w) >= UPSCALE_THRESHOLD_PX:
        return UpscaleResult(image=img.copy(), was_upscaled=False)

    model = _get_model()
    if model is None:
        return UpscaleResult(image=img.copy(), was_upscaled=False)

    upscaled = _run_esrgan(model, img)
    return UpscaleResult(image=upscaled, was_upscaled=True)


def _get_model():
    global _model, _model_loaded
    if _model_loaded:
        return _model
    _model_loaded = True
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        arch = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        weights_path = (
            Path(__file__).parent.parent.parent / "weights" / "RealESRGAN_x4plus.pth"
        )
        _model = RealESRGANer(
            scale=4,
            model_path=str(weights_path),
            model=arch,
            tile=256,
            tile_pad=10,
            pre_pad=0,
            half=False,
        )
    except Exception as e:
        logger.warning("Real-ESRGAN unavailable: %s", e)
        _model = None
    return _model


def _run_esrgan(model, img: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output, _ = model.enhance(rgb, outscale=4)
    return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)


def reset_model_cache() -> None:
    global _model, _model_loaded
    _model = None
    _model_loaded = False
