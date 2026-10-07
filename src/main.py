"""Command line interface: compress, decompress and bench.

    python -m src.main compress   input output [--model cts|ctw] [--depth 48]
    python -m src.main decompress input output
    python -m src.main bench      file [file ...] [--depth 48] [--limit BYTES]

`python src/main.py ...` works too.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # run as a plain script: make `src` importable
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cts_core.codec import MODELS, code_length, compress, decompress  # noqa: E402

DEFAULT_DEPTH = 48  # the paper's CTW48 / CTS48: six bytes of context


def cmd_compress(args) -> int:
    data = Path(args.input).read_bytes()
    start = time.perf_counter()
    blob = compress(data, args.model, args.depth)
    seconds = time.perf_counter() - start
    Path(args.output).write_bytes(blob)
    ratio = len(blob) / len(data) if data else float("nan")
    print(f"{args.input}: {len(data)} -> {len(blob)} bytes ({ratio:.3f}), "
          f"{args.model} depth {args.depth}, {seconds:.1f}s", file=sys.stderr)
    return 0


def cmd_decompress(args) -> int:
    blob = Path(args.input).read_bytes()
    try:
        data = decompress(blob)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    Path(args.output).write_bytes(data)
    print(f"{args.input}: {len(blob)} -> {len(data)} bytes", file=sys.stderr)
    return 0


def cmd_bench(args) -> int:
    """Average bits per byte of CTW and CTS, from the ideal code length -log2 P(data)."""
    print(f"depth {args.depth}, bits per byte (ideal code length)")
    print(f"{'file':<10}{'bytes':>9}{'CTW':>9}{'CTS':>9}{'CTW s':>9}{'CTS s':>9}")
    total_bytes = 0
    total_bits = {"ctw": 0.0, "cts": 0.0}
    for name in args.files:
        data = Path(name).read_bytes()
        if args.limit:
            data = data[:args.limit]
        if not data:
            print(f"{Path(name).name:<10}  (empty, skipped)")
            continue
        bpb, seconds = {}, {}
        for model in ("ctw", "cts"):
            start = time.perf_counter()
            bits = code_length(data, model, args.depth)
            seconds[model] = time.perf_counter() - start
            bpb[model] = bits / len(data)
            total_bits[model] += bits
        total_bytes += len(data)
        print(f"{Path(name).name:<10}{len(data):>9}{bpb['ctw']:>9.3f}{bpb['cts']:>9.3f}"
              f"{seconds['ctw']:>9.1f}{seconds['cts']:>9.1f}", flush=True)
    if total_bytes and len(args.files) > 1:
        print(f"{'weighted':<10}{total_bytes:>9}{total_bits['ctw'] / total_bytes:>9.3f}"
              f"{total_bits['cts'] / total_bytes:>9.3f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main", description="Context Tree Switching compressor")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("compress", help="compress a file")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--model", choices=sorted(MODELS), default="cts")
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="context depth in bits")
    p.set_defaults(run=cmd_compress)

    p = commands.add_parser("decompress", help="restore a file made by compress")
    p.add_argument("input")
    p.add_argument("output")
    p.set_defaults(run=cmd_decompress)

    p = commands.add_parser("bench", help="bits per byte of CTW and CTS on files")
    p.add_argument("files", nargs="+")
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="context depth in bits")
    p.add_argument("--limit", type=int, default=0, help="only use the first LIMIT bytes of each file")
    p.set_defaults(run=cmd_bench)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.run(args)
    except ValueError as error:  # e.g. a depth outside 0..255
        print(f"error: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
