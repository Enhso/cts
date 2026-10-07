import math
import random
import subprocess
import sys
from pathlib import Path

import pytest

from src.cts_core.codec import HEADER, MAGIC, code_length, compress, decompress
from src.main import main

MAIN_PY = Path(__file__).resolve().parent.parent / "src" / "main.py"

TEXT = (
    b"Context tree switching mixes over a strictly larger class of models than "
    b"context tree weighting, without increasing the time or space needed. "
    b"Instead of weighting two alternatives at each node it lets the better one "
    b"change over time, so a deep context can take over from a shallow one once "
    b"it has seen enough data. The quick brown fox jumps over the lazy dog. "
) * 3

ALL_BYTES = bytes(range(256))
RANDOM_BYTES = random.Random(0).randbytes(400)

CASES = {
    "empty": b"",
    "one zero byte": b"\x00",
    "one 0xff byte": b"\xff",
    "one letter": b"a",
    "text": TEXT,
    "random bytes": RANDOM_BYTES,
    "every byte value": ALL_BYTES,
    "long run": b"\x00" * 3000,
    "two-byte pattern": b"\xaa\x55" * 400,
}


@pytest.mark.parametrize("model", ["cts", "ctw"])
@pytest.mark.parametrize("depth", [0, 3, 16])
@pytest.mark.parametrize("name", CASES)
def test_round_trip(name, depth, model):
    data = CASES[name]
    assert decompress(compress(data, model, depth)) == data


@pytest.mark.parametrize("model", ["cts", "ctw"])
def test_round_trip_at_the_papers_depth(model):
    for data in (b"", b"x", TEXT[:300], RANDOM_BYTES[:100]):
        assert decompress(compress(data, model, 48)) == data


def test_compression_is_deterministic():
    assert compress(TEXT, "cts", 16) == compress(TEXT, "cts", 16)


@pytest.mark.parametrize("model", ["cts", "ctw"])
def test_output_size_matches_the_ideal_code_length(model):
    for data in (TEXT, RANDOM_BYTES):
        blob = compress(data, model, 16)
        body_bits = (len(blob) - HEADER.size) * 8
        ideal = code_length(data, model, 16)
        # Arithmetic coding costs under 3 bits over the ideal (Eq. 1) plus byte padding.
        assert ideal - 1 <= body_bits <= ideal + 3 + 7


def test_text_compresses_and_random_data_does_not():
    # Measured: about 0.51 of the original at depth 16. With only 1 KB and no
    # enhancements the KT prior keeps this from going much lower.
    with_context = len(compress(TEXT, "cts", 16))
    assert with_context < 0.6 * len(TEXT)
    assert with_context < 0.7 * len(compress(TEXT, "cts", 0))  # context is what helps
    assert len(compress(b"\x00" * 3000, "cts", 16)) < 0.05 * 3000
    assert len(compress(RANDOM_BYTES, "cts", 16)) < len(RANDOM_BYTES) * 1.03 + HEADER.size


def test_code_length_of_empty_input_is_zero():
    assert code_length(b"", "cts", 8) == 0
    assert code_length(b"", "ctw", 8) == 0


def test_code_length_is_minus_log2_of_probability_for_one_byte():
    # Depth 0 is a plain KT estimator over the 8 bits of 0xff: P = (2*8-1)!! / (2^8 * 8!).
    probability = math.prod(range(1, 16, 2)) / (2 ** 8 * math.factorial(8))
    assert code_length(b"\xff", "cts", 0) == pytest.approx(-math.log2(probability), rel=1e-12)


def test_decompress_rejects_bad_input():
    good = compress(b"hello", "cts", 4)
    with pytest.raises(ValueError):
        decompress(good[: HEADER.size - 1])
    with pytest.raises(ValueError):
        decompress(b"NOPE" + good[4:])
    with pytest.raises(ValueError):
        decompress(MAGIC + bytes([9]) + good[5:])  # unknown model id


def test_compress_rejects_bad_arguments():
    with pytest.raises(ValueError):
        compress(b"x", "cts", 256)
    with pytest.raises(ValueError):
        compress(b"x", "cts", -1)
    with pytest.raises(ValueError):
        compress(b"x", "lz77", 8)


@pytest.mark.parametrize("model", ["cts", "ctw"])
def test_cli_compress_then_decompress_restores_the_file(tmp_path, model):
    original, packed, restored = tmp_path / "in.txt", tmp_path / "in.cts", tmp_path / "out.txt"
    original.write_bytes(TEXT[:400])
    assert main(["compress", str(original), str(packed), "--model", model, "--depth", "8"]) == 0
    assert packed.stat().st_size < 400
    assert main(["decompress", str(packed), str(restored)]) == 0
    assert restored.read_bytes() == original.read_bytes()


def test_cli_handles_an_empty_file(tmp_path):
    empty, packed, restored = tmp_path / "empty", tmp_path / "empty.cts", tmp_path / "empty.out"
    empty.write_bytes(b"")
    assert main(["compress", str(empty), str(packed), "--depth", "4"]) == 0
    assert main(["decompress", str(packed), str(restored)]) == 0
    assert restored.read_bytes() == b""


def test_cli_reports_errors_instead_of_crashing(tmp_path):
    junk = tmp_path / "junk"
    junk.write_bytes(b"this is not a compressed file")
    assert main(["decompress", str(junk), str(tmp_path / "out")]) == 1
    assert main(["compress", str(tmp_path / "missing"), str(tmp_path / "out")]) == 1
    assert main(["compress", str(junk), str(tmp_path / "out"), "--depth", "999"]) == 1


def test_cli_bench_reports_bits_per_byte(tmp_path, capsys):
    sample = tmp_path / "sample"
    sample.write_bytes(TEXT[:300])
    assert main(["bench", str(sample), "--depth", "8"]) == 0
    row = capsys.readouterr().out.splitlines()[-1].split()
    assert row[0] == "sample" and row[1] == "300"
    ctw, cts = float(row[2]), float(row[3])
    assert ctw == pytest.approx(code_length(TEXT[:300], "ctw", 8) / 300, abs=1e-3)
    assert cts == pytest.approx(code_length(TEXT[:300], "cts", 8) / 300, abs=1e-3)


def test_cli_bench_limit_uses_only_a_prefix(tmp_path, capsys):
    sample = tmp_path / "sample"
    sample.write_bytes(TEXT)
    assert main(["bench", str(sample), "--depth", "4", "--limit", "100"]) == 0
    assert capsys.readouterr().out.splitlines()[-1].split()[1] == "100"


def test_script_runs_in_isolated_mode_from_another_directory(tmp_path):
    original, packed, restored = tmp_path / "a", tmp_path / "a.cts", tmp_path / "a.out"
    original.write_bytes(b"hello hello hello hello")
    run = lambda *args: subprocess.run(
        [sys.executable, "-I", str(MAIN_PY), *args], cwd=tmp_path, capture_output=True, text=True)
    assert run("compress", str(original), str(packed), "--depth", "6").returncode == 0
    assert run("decompress", str(packed), str(restored)).returncode == 0
    assert restored.read_bytes() == original.read_bytes()
