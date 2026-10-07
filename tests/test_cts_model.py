import itertools
import math
import random

import pytest

from src.cts_core import cts_model
from src.cts_core.cts_model import CTSModel, CTWModel
from src.cts_core.kt_estimator import KTEstimator
from tests.naive_reference import naive_ctw, naive_cts, precise_cts

MODELS = [CTWModel, CTSModel]
NAIVE = {CTWModel: lambda bits, depth: math.log(naive_ctw(bits, depth)),
         CTSModel: lambda bits, depth: math.log(naive_cts(bits, depth))}


def random_bits(n, seed, p_one=0.5):
    rng = random.Random(seed)
    return [1 if rng.random() < p_one else 0 for _ in range(n)]


def run(model_cls, depth, bits):
    model = model_cls(depth)
    for bit in bits:
        model.update(bit)
    return model


def mismatches(model_cls, naive_log_joint, depth, max_len, tol=1e-9):
    """Every binary sequence of length 1..max_len on which the incremental model
    and the naive recursion disagree about ln P(x_1:n)."""
    bad = []
    for n in range(1, max_len + 1):
        for bits in itertools.product((0, 1), repeat=n):
            bits = list(bits)
            if abs(run(model_cls, depth, bits).log_joint - naive_log_joint(bits, depth)) > tol:
                bad.append(bits)
    return bad


@pytest.mark.parametrize("model_cls", MODELS)
def test_depth_zero_reduces_to_kt(model_cls):
    for seed in range(5):
        model, kt = model_cls(0), KTEstimator()
        for bit in random_bits(200, seed, p_one=0.2 + 0.15 * seed):
            assert model.log_prob(bit) == pytest.approx(kt.log_prob(bit), abs=1e-12)
            model.update(bit)
            kt.update(bit)
            assert model.log_joint == pytest.approx(kt.log_joint, abs=1e-10)


@pytest.mark.parametrize("model_cls", MODELS)
@pytest.mark.parametrize("depth", [0, 1, 2, 3])
def test_matches_naive_recursion_on_all_short_sequences(model_cls, depth):
    assert mismatches(model_cls, NAIVE[model_cls], depth, max_len=10) == []


@pytest.mark.parametrize("model_cls", MODELS)
@pytest.mark.parametrize("depth", [0, 1, 2, 5])
def test_probabilities_sum_to_one_at_every_step(model_cls, depth):
    model = model_cls(depth)
    for bit in random_bits(300, seed=depth, p_one=0.3):
        p0, p1 = math.exp(model.log_prob(0)), math.exp(model.log_prob(1))
        assert p0 + p1 == pytest.approx(1.0, abs=1e-12)
        assert model.predict() == pytest.approx(p1, abs=1e-15)
        model.update(bit)


@pytest.mark.parametrize("model_cls", MODELS)
def test_log_prob_is_read_only_and_agrees_with_update(model_cls):
    bits = random_bits(150, seed=7)
    probed, plain = model_cls(3), model_cls(3)
    for bit in bits:
        predicted = probed.log_prob(bit)
        probed.log_prob(1 - bit)
        probed.predict()
        assert probed.update(bit) == pytest.approx(predicted, abs=1e-12)
        plain.update(bit)
        assert probed.log_joint == plain.log_joint  # probing left no trace


@pytest.mark.parametrize("model_cls", MODELS)
def test_code_length_is_minus_log2_of_the_joint_probability(model_cls):
    bits = random_bits(9, seed=3)
    model = model_cls(2)
    summed = 0.0
    for bit in bits:
        summed += -math.log2(math.exp(model.log_prob(bit)))
        model.update(bit)
    expected = -NAIVE[model_cls](bits, 2) / math.log(2)
    assert model.code_length == pytest.approx(expected, rel=1e-12)
    assert summed == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("model_cls", MODELS)
@pytest.mark.parametrize("depth", [1, 4, 8])
@pytest.mark.parametrize("constant", [0, 1])
def test_constant_sequence_costs_only_logarithmically(model_cls, depth, constant):
    # For S = {epsilon} (one leaf, depth 0, Gamma_D(S) = 1, -log2 Pr(x|S,theta) = 0)
    # the paper bounds the code length by
    #   CTW, Eq. 15: Gamma + gamma(n)                 = 1 + (1/2) log2 n + 1
    #   CTS, Eq. 17: Gamma + (d(S)+1) log2 n + gamma(n) = 1 + log2 n + (1/2) log2 n + 1
    model = model_cls(depth)
    lengths = {}
    for n in range(1, 8193):
        model.update(constant)
        if n in (256, 512, 8192):
            lengths[n] = model.code_length
            gamma = 0.5 * math.log2(n) + 1
            bound = 1 + gamma + (math.log2(n) if model_cls is CTSModel else 0)
            assert lengths[n] < bound
    assert lengths[512] - lengths[256] < 1.0  # doubling the data costs under a bit
    assert lengths[8192] < 0.005 * 8192


def text_bits(n):
    from src.utils.io import iter_bits
    from tests.test_integration import TEXT
    return list(iter_bits(TEXT))[:n]


@pytest.mark.parametrize("bits", [random_bits(400, seed=21), text_bits(400)], ids=["random", "text"])
def test_log_space_agrees_with_high_precision_arithmetic_on_longer_sequences(bits):
    depth = 6
    expected_ctw = math.log(naive_ctw(bits, depth))
    expected_cts = precise_cts(bits, depth)
    assert run(CTWModel, depth, bits).log_joint == pytest.approx(expected_ctw, abs=1e-9)
    assert run(CTSModel, depth, bits).log_joint == pytest.approx(expected_cts, abs=1e-9)


def test_precise_cts_agrees_with_the_naive_recursion():
    # Ties the high-precision incremental version to Eq. 16, where Eq. 16 is feasible.
    for depth in (1, 2, 3):
        for bits in itertools.islice(itertools.product((0, 1), repeat=9), 0, 512, 7):
            assert precise_cts(list(bits), depth) == pytest.approx(math.log(naive_cts(list(bits), depth)), abs=1e-12)


@pytest.mark.parametrize("model_cls", MODELS)
def test_long_sequences_do_not_underflow(model_cls):
    n = 100_000
    model = run(model_cls, 2, random_bits(n, seed=11))
    assert math.exp(model.log_joint) == 0.0  # the plain probability is long gone ...
    assert math.isfinite(model.log_joint)  # ... but the log is exact
    assert 0.99 * n < model.code_length < 1.01 * n  # random bits cost about one bit each


def test_leaf_counts_match_direct_counting():
    depth, bits = 3, random_bits(500, seed=5, p_one=0.4)
    model = run(CTSModel, depth, bits)
    expected = {}
    history = [0] * depth + bits
    for t in range(depth, len(history)):
        context = tuple(history[t - j] for j in range(1, depth + 1))  # most recent first
        counts = expected.setdefault(context, [0, 0])
        counts[history[t]] += 1
    for context, (zeros, ones) in expected.items():
        leaf = model.tree.get_nodes_for_context(list(context))[0]
        assert (leaf.count_0, leaf.count_1) == (zeros, ones)
    root = model.tree.root
    assert root.count_0 + root.count_1 == len(bits)


def test_cts_with_zero_switching_rate_is_ctw(monkeypatch):
    monkeypatch.setattr(cts_model, "switching_rate", lambda t: 0.0)
    for seed in range(4):
        bits = random_bits(200, seed)
        assert run(CTSModel, 4, bits).log_joint == pytest.approx(run(CTWModel, 4, bits).log_joint, abs=1e-10)


def test_cts_differs_from_ctw_with_the_paper_schedule():
    bits = [0] * 50 + [1] * 50
    assert run(CTSModel, 3, bits).log_joint != pytest.approx(run(CTWModel, 3, bits).log_joint, abs=1e-3)


# The tests below show that the naive-vs-incremental comparison has teeth: with a
# deliberately wrong parameter on one side, it must report mismatches.

PERTURBED_RATES = {
    "rate scaled by 0.9": lambda t: 0.9 / t,
    "off by one in time": lambda t: 1.0 / (t + 1),
    "rate halved": lambda t: 0.5 / t,
}


@pytest.mark.parametrize("name", PERTURBED_RATES)
def test_planted_switching_rate_bug_in_the_model_is_caught(monkeypatch, name):
    assert mismatches(CTSModel, NAIVE[CTSModel], depth=2, max_len=7) == []  # control
    monkeypatch.setattr(cts_model, "switching_rate", PERTURBED_RATES[name])
    bad = mismatches(CTSModel, NAIVE[CTSModel], depth=2, max_len=7)
    assert len(bad) > 100  # out of 254 sequences


@pytest.mark.parametrize("name", PERTURBED_RATES)
def test_planted_switching_rate_bug_in_the_reference_is_caught(name):
    wrong = lambda bits, depth: math.log(naive_cts(bits, depth, rate=PERTURBED_RATES[name]))
    assert len(mismatches(CTSModel, wrong, depth=2, max_len=7)) > 100


def test_planted_ctw_weight_bug_is_caught(monkeypatch):
    assert mismatches(CTWModel, NAIVE[CTWModel], depth=2, max_len=7) == []  # control
    monkeypatch.setattr(cts_model, "LOG_HALF", math.log(0.6))
    assert len(mismatches(CTWModel, NAIVE[CTWModel], depth=2, max_len=7)) > 100


@pytest.mark.parametrize("model_cls", MODELS)
def test_rejects_non_bits(model_cls):
    model = model_cls(2)
    with pytest.raises(ValueError):
        model.update(2)
    with pytest.raises(ValueError):
        model.log_prob(-1)
