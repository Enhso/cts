import math
import random

import pytest

from src.cts_core.codec import HEADER, MAGIC, code_length, compress, decompress

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
