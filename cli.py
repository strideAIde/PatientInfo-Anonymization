from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anonymizer.preprocessing.upscale as _upscale_module
from anonymizer.pipeline import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anonymizer",
        description="Blur patient name and ID fields in medical report images.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        metavar="INPUT",
        help="One or more input image paths (JPEG recommended).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="Directory where redacted images are saved.",
    )
    parser.add_argument(
        "--no-upscale",
        action="store_true",
        default=False,
        help="Disable Real-ESRGAN upscaling (faster, lower OCR quality on small images).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress per-file progress output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.no_upscale:
        _upscale_module.UPSCALE_ENABLED = False

    args.output_dir.mkdir(parents=True, exist_ok=True)

    failed = 0
    for input_path in args.inputs:
        output_path = args.output_dir / input_path.name
        try:
            result = run(input_path, output_path)
            if not args.quiet:
                print(
                    f"{input_path.name}: {result.regions_redacted} region(s) redacted"
                    f" → {output_path}"
                )
        except Exception as exc:
            print(f"ERROR {input_path.name}: {exc}", file=sys.stderr)
            failed += 1

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
