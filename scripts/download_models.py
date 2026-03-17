from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

WEIGHTS_DIR = Path(__file__).parent.parent / "weights"

REALESRGAN_FILENAME = "RealESRGAN_x4plus.pth"
REALESRGAN_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
    "RealESRGAN_x4plus.pth"
)

OCR_LANGUAGES = ["en"]


def download_realesrgan(dest: Path, force: bool) -> None:
    target = dest / REALESRGAN_FILENAME
    if target.exists() and not force:
        print(f"[skip] {target} already exists (use --force to re-download)")
        return
    print(f"[downloading] Real-ESRGAN weights → {target}")
    try:
        import urllib.request

        dest.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(REALESRGAN_URL, target)
        print(f"[done] {target} ({target.stat().st_size // 1024 // 1024} MB)")
    except Exception as exc:
        print(f"[error] Failed to download Real-ESRGAN weights: {exc}", file=sys.stderr)
        sys.exit(1)


def download_ocr_model(force: bool) -> None:
    print("[downloading] EasyOCR models (English)")
    try:
        import easyocr
    except ImportError as exc:
        print(
            f"[error] easyocr not installed: {exc}\nRun: pip install easyocr",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        easyocr.Reader(OCR_LANGUAGES, gpu=False)
        print("[done] EasyOCR models cached")
    except Exception as exc:
        print(f"[error] Failed to initialise EasyOCR: {exc}", file=sys.stderr)
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download model weights required by the anonymizer pipeline."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Re-download even if files already exist.",
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        default=False,
        help="Skip downloading the OCR model.",
    )
    parser.add_argument(
        "--skip-esrgan",
        action="store_true",
        default=False,
        help="Skip downloading the Real-ESRGAN weights.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.skip_esrgan:
        download_realesrgan(WEIGHTS_DIR, args.force)

    if not args.skip_ocr:
        download_ocr_model(args.force)

    print("[all done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
