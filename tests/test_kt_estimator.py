import itertools
import math
from fractions import Fraction

import pytest

from src.cts_core.kt_estimator import KTEstimator, kt_prob


def double_factorial_odd(k):
    """(2k-1)!! = 1 * 3 * 5 * ... * (2k-1), with (-1)!! = 1."""
    result = 1
    for j in range(1, k + 1):
        result *= 2 * j - 1
    return result


def kt_closed_form(zeros, ones):
    """xi_KT = Gamma(a+1/2) Gamma(b+1/2) / (pi (a+b)!), written with exact integers.

    Using Gamma(k+1/2) = sqrt(pi) (2k-1)!! / 2^k this becomes
    (2a-1)!! (2b-1)!! / (2^(a+b) (a+b)!).
    """
    numerator = double_factorial_odd(zeros) * double_factorial_odd(ones)
    return Fraction(numerator, 2 ** (zeros + ones) * math.factorial(zeros + ones))


def sequential_product(bits):
    """Chain rule over Eq. 7 in exact arithmetic."""
    zeros = ones = 0
    prob = Fraction(1)
    for bit in bits:
        seen = ones if bit else zeros
        prob *= Fraction(2 * seen + 1, 2 * (zeros + ones + 1))
        ones += bit
        zeros += 1 - bit
    return prob


def test_known_small_values():
    assert kt_prob(0, 0, 0) == 0.5
    assert kt_prob(1, 0, 0) == 0.5
    assert kt_prob(0, 1, 0) == 0.75
    assert kt_prob(1, 1, 0) == 0.25
    assert sequential_product([0, 1]) == Fraction(1, 8)


def test_sequential_product_equals_closed_form_for_every_sequence():
    for n in range(0, 11):
        for bits in itertools.product((0, 1), repeat=n):
            ones = sum(bits)
            assert sequential_product(bits) == kt_closed_form(n - ones, ones)


def test_joint_depends_only_on_counts():
    expected = kt_closed_form(3, 2)
    for bits in set(itertools.permutations([0, 0, 0, 1, 1])):
        assert sequential_product(bits) == expected


def test_closed_form_matches_the_integral_definition():
    # Eq. 6: integral of theta^b (1-theta)^a / (pi sqrt(theta (1-theta))).
    # Substituting theta = sin^2(phi) removes the endpoint singularities:
    # (2/pi) * integral over [0, pi/2] of sin(phi)^(2b) cos(phi)^(2a).
    steps = 20000
    for zeros, ones in [(0, 0), (1, 0), (0, 3), (2, 5), (7, 7), (10, 1)]:
        h = (math.pi / 2) / steps
        total = sum(
            math.sin((i + 0.5) * h) ** (2 * ones) * math.cos((i + 0.5) * h) ** (2 * zeros)
            for i in range(steps)
        )
        integral = 2 / math.pi * total * h
        assert integral == pytest.approx(float(kt_closed_form(zeros, ones)), rel=1e-8)


def test_gamma_function_form():
    for zeros, ones in [(0, 0), (4, 9), (30, 2), (100, 100)]:
        log_gamma_form = (
            math.lgamma(zeros + 0.5) + math.lgamma(ones + 0.5)
            - math.lgamma(zeros + ones + 1) - math.log(math.pi)
        )
        exact = kt_closed_form(zeros, ones)
        # math.log accepts the big Fraction exactly; avoid float underflow of the ratio.
        exact_log = math.log(exact.numerator) - math.log(exact.denominator)
        assert log_gamma_form == pytest.approx(exact_log, abs=1e-9)


def test_estimator_joint_is_closed_form_and_product_of_predictions():
    bits = [0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0]
    estimator = KTEstimator()
    product = 1.0
    for bit in bits:
        p1 = estimator.predict()
        product *= p1 if bit else 1 - p1
        estimator.update(bit)
    ones = sum(bits)
    exact = kt_closed_form(len(bits) - ones, ones)
    assert math.exp(estimator.log_joint) == pytest.approx(float(exact), rel=1e-12)
    assert product == pytest.approx(float(exact), rel=1e-12)


def test_code_length_is_minus_log2_of_joint_probability():
    bits = [1, 1, 0, 1, 0, 0, 0, 1]
    estimator = KTEstimator()
    summed = 0.0
    for bit in bits:
        summed += -estimator.log_prob(bit) / math.log(2)
        estimator.update(bit)
    ones = sum(bits)
    exact = kt_closed_form(len(bits) - ones, ones)
    expected = -math.log2(exact.numerator / exact.denominator)
    assert estimator.code_length == pytest.approx(expected, rel=1e-12)
    assert summed == pytest.approx(expected, rel=1e-12)


def test_probabilities_sum_to_one():
    estimator = KTEstimator()
    for bit in [0, 1, 1, 0, 0, 0, 1]:
        assert math.exp(estimator.log_prob(0)) + math.exp(estimator.log_prob(1)) == pytest.approx(1.0, abs=1e-15)
        estimator.update(bit)


def test_update_rejects_non_bits():
    with pytest.raises(ValueError):
        KTEstimator().update(2)
