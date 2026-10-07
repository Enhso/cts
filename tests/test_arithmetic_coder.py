import math
import random

import pytest

from src.cts_core.arithmetic_coder import PROB_ONE, Decoder, Encoder, quantize


def encode_all(bits, probs):
    encoder = Encoder()
    for bit, p1 in zip(bits, probs):
        encoder.encode(bit, p1)
    return encoder, encoder.finish()


def decode_all(data, probs):
    decoder = Decoder(data)
    return [decoder.decode(p1) for p1 in probs]


def ideal_bits(bits, probs):
    return sum(-math.log2((p1 if bit else PROB_ONE - p1) / PROB_ONE) for bit, p1 in zip(bits, probs))


def test_quantize_clamps_away_from_certainty():
    assert quantize(0.0) == 1
    assert quantize(1.0) == PROB_ONE - 1
    assert quantize(0.5) == PROB_ONE // 2
    assert quantize(1e-12) == 1


@pytest.mark.parametrize("seed", range(5))
def test_round_trip_with_random_probabilities(seed):
    rng = random.Random(seed)
    n = 20000
    probs = [rng.choice([1, 2, PROB_ONE // 2, PROB_ONE - 2, PROB_ONE - 1, rng.randrange(1, PROB_ONE)]) for _ in range(n)]
    # Mostly bits the model expects, but also surprises, including at p = 1/65536.
    bits = [1 if rng.random() < p / PROB_ONE else 0 for p in probs]
    bits[::97] = [1 - b for b in bits[::97]]
    _, data = encode_all(bits, probs)
    assert decode_all(data, probs) == bits


def test_round_trip_of_an_empty_stream():
    _, data = encode_all([], [])
    assert decode_all(data, []) == []


def test_round_trip_of_single_bits():
    for bit in (0, 1):
        for p1 in (1, PROB_ONE // 2, PROB_ONE - 1):
            _, data = encode_all([bit], [p1])
            assert decode_all(data, [p1]) == [bit]


@pytest.mark.parametrize("p_one", [0.5, 0.1, 0.999])
def test_output_length_is_within_a_few_bits_of_the_ideal(p_one):
    # Eq. 1: l(x) < ceil(-log2 mu(x)) + 2, here for the quantized model probabilities.
    rng = random.Random(1)
    n = 30000
    probs = [quantize(p_one)] * n
    bits = [1 if rng.random() < p_one else 0 for _ in range(n)]
    encoder, data = encode_all(bits, probs)
    assert len(encoder.bits) <= math.ceil(ideal_bits(bits, probs)) + 2
    assert decode_all(data, probs) == bits


def test_a_long_run_of_the_likely_bit_is_nearly_free():
    n = 100000
    probs = [PROB_ONE - 1] * n
    _, data = encode_all([1] * n, probs)
    assert len(data) < 20  # 100000 * 2.2e-5 bits is about 2.2 bits, plus the flush
    assert decode_all(data, probs) == [1] * n
