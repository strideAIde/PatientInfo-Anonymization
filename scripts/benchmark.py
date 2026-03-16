from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure anonymizer throughput on a set of images."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        metavar="INPUT",
        help="Image files to benchmark.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="Directory for redacted output images.",
    )
    parser.add_argument(
        "--no-upscale",
        action="store_true",
        default=False,
        help="Disable Real-ESRGAN upscaling during benchmark.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        metavar="N",
        help="Number of warm-up images to process before timing starts (default: 1).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import anonymizer.preprocessing.upscale as _upscale_module
    from anonymizer.pipeline import run

    if args.no_upscale:
        _upscale_module.UPSCALE_ENABLED = False

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_paths = list(args.inputs)
    warmup_paths = all_paths[: args.warmup]
    bench_paths = all_paths[args.warmup :]

    if warmup_paths:
        print(f"Warming up on {len(warmup_paths)} image(s)...")
        for path in warmup_paths:
            out = args.output_dir / path.name
            try:
                run(path, out)
            except Exception as exc:
                print(f"  [warn] warm-up skipped {path.name}: {exc}", file=sys.stderr)

    if not bench_paths:
        print("No images left for benchmarking after warm-up.", file=sys.stderr)
        return 1

    print(f"Benchmarking {len(bench_paths)} image(s)...")
    times: list[float] = []
    errors = 0

    for path in bench_paths:
        out = args.output_dir / path.name
        t0 = time.perf_counter()
        try:
            run(path, out)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            print(f"  {path.name}: {elapsed:.2f}s")
        except Exception as exc:
            print(f"  [error] {path.name}: {exc}", file=sys.stderr)
            errors += 1

    if not times:
        print("No successful runs to report.", file=sys.stderr)
        return 1

    total = sum(times)
    mean = statistics.mean(times)
    median = statistics.median(times)
    throughput = len(times) / total * 60

    print()
    print("=" * 40)
    print(f"Images processed : {len(times)}")
    print(f"Errors           : {errors}")
    print(f"Total time       : {total:.2f}s")
    print(f"Mean / image     : {mean:.2f}s")
    print(f"Median / image   : {median:.2f}s")
    print(f"Throughput       : {throughput:.1f} img/min")
    print("=" * 40)

    target = 10.0
    if throughput >= target:
        print(f"PASS  throughput >= {target} img/min")
    else:
        print(f"WARN  throughput {throughput:.1f} img/min < {target} img/min target")

    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
