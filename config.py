# Central configuration for the PatientInfo-Anonymization pipeline.
# All tunables live here — change values here rather than in individual modules.

# ── Preprocessing ──────────────────────────────────────────────────────────────
CLAHE_CLIP_LIMIT: float = 3.0
CLAHE_TILE_GRID: tuple[int, int] = (8, 8)
DENOISE_H: int = 3           # fastNlMeansDenoisingColored luminance strength
DENOISE_H_COLOR: int = 3     # fastNlMeansDenoisingColored color strength
SKEW_CORRECTION_MIN_ANGLE_DEG: float = 1.0  # skip correction if skew < this

# ── Upscaling (Real-ESRGAN) ────────────────────────────────────────────────────
# Enabled by default but only applied when image is below threshold.
# Modern smartphone photos (>1200px shortest edge) are skipped automatically.
UPSCALE_ENABLED: bool = True
UPSCALE_THRESHOLD_PX: int = 1200   # apply upscaling if shortest edge < this

# ── OCR ────────────────────────────────────────────────────────────────────────
OCR_LANGUAGES: list[str] = ["en"]
OCR_CONFIDENCE_THRESHOLD: float = 0.3
# Set to True to force GPU, False to force CPU, None to auto-detect
OCR_USE_GPU: bool | None = None
# Vertical tolerance (px) for grouping OCR tokens into the same text line
LINE_Y_TOLERANCE_PX: int = 10

# ── PII Detection ──────────────────────────────────────────────────────────────
# Tokens on a Name line that signal the name has ended — never blur these
NAME_STOP_WORDS: list[str] = [
    "KIER", "PATIENT", "GENDER", "CONTACT", "DOB",
    "AGE", "CONSULTATION", "REPORT", "DIABETIC", "HBA1C",
]

# ── Redaction ──────────────────────────────────────────────────────────────────
BLUR_KERNEL_SIZE: int = 51    # must be odd
BLUR_PASSES: int = 2          # number of Gaussian blur passes (more = more opaque)
BLUR_PADDING_PX: int = 8      # padding added around each bounding box before blur
