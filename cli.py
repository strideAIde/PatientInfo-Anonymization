from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anonymizer.preprocessing.upscale as _upscale_module
from anonymizer.pipeline import run

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
_DEFAULT_INPUT_DIR = Path(__file__).parent / "input"
_DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"


def _collect_images(inputs: list[Path]) -> list[Path]:
    images: list[Path] = []
    for p in inputs:
        if p.is_dir():
            images.extend(
                f for f in sorted(p.iterdir()) if f.suffix.lower() in _IMAGE_SUFFIXES
            )
        else:
            images.append(p)
    return images


def _process_one(input_path: Path, output_path: Path) -> tuple[Path, int | None, Exception | None]:
    try:
        result = run(input_path, output_path)
        return input_path, result.regions_redacted, None
    except Exception as exc:
        return input_path, None, exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anonymizer",
        description="Blur patient name and ID fields in medical report images.",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        metavar="INPUT",
        help=f"Image paths or directories to process (default: {_DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        metavar="DIR",
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Directory where redacted images are saved (default: {_DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel workers for batch processing (default: 1).",
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
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging verbosity (default: WARNING).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(name)s: %(message)s")

    if args.no_upscale:
        _upscale_module.UPSCALE_ENABLED = False

    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_sources = args.inputs if args.inputs else [_DEFAULT_INPUT_DIR]
    images = _collect_images(input_sources)
    if not images:
        print("No images found.", file=sys.stderr)
        return 2

    if not args.quiet:
        print(f"Processing {len(images)} image(s) with {args.workers} worker(s)...")

    tasks = [(p, args.output_dir / p.name) for p in images]
    failed = 0

    if args.workers == 1:
        for input_path, output_path in tasks:
            _, regions, exc = _process_one(input_path, output_path)
            if exc:
                print(f"ERROR {input_path.name}: {exc}", file=sys.stderr)
                failed += 1
            elif not args.quiet:
                print(f"{input_path.name}: {regions} region(s) redacted -> {output_path}")
    else:
        futures = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for input_path, output_path in tasks:
                fut = pool.submit(_process_one, input_path, output_path)
                futures[fut] = input_path
            for fut in as_completed(futures):
                input_path, regions, exc = fut.result()
                if exc:
                    print(f"ERROR {input_path.name}: {exc}", file=sys.stderr)
                    failed += 1
                elif not args.quiet:
                    print(f"{input_path.name}: {regions} region(s) redacted -> {args.output_dir / input_path.name}")

    total = len(images)
    if not args.quiet:
        print(f"Done: {total - failed}/{total} succeeded.")

    return 0 if failed == 0 else (1 if failed < total else 2)


if __name__ == "__main__":
    sys.exit(main())
