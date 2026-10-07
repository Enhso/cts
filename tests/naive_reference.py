"""Direct, non-incremental versions of the paper's recursive definitions.

These exist only to cross-check the incremental models in src/. They are slow
(CTS sums over all 2^n_c switch sequences, Eq. 16) and deliberately share no
code with src/. Bits are x_1 ... x_n in a Python list, so x_t is bits[t - 1].
Positions before the start are treated as 0, matching the zero padding in
src/cts_core/cts_model.py.

A context c is a tuple of bits, most recent first, i.e. (x_{t-1}, x_{t-2}, ...).
Its children are c + (0,) and c + (1,): the same context looking one bit
further into the past (the paper writes these 0c and 1c).
"""
from __future__ import annotations

import itertools
from fractions import Fraction


def kt_prefixes(seq):
    """xi_KT of every prefix of seq, by the chain rule over Eq. 7 (index k = first k bits)."""
    zeros = ones = 0
    probs = [Fraction(1)]
    for bit in seq:
        seen = ones if bit else zeros
        probs.append(probs[-1] * Fraction(2 * seen + 1, 2 * (zeros + ones + 1)))
        ones += bit
        zeros += 1 - bit
    return probs


def context_at(bits, t, length):
    """The first `length` bits of phi(x_<t) = x_{t-1} x_{t-2} ..., for 1-based t."""
    return tuple(bits[t - j - 1] if t - j >= 1 else 0 for j in range(1, length + 1))


def times_in_context(bits, c, n):
    """The times t <= n at which context c applies: t_c(1) < t_c(2) < ... (1-based)."""
    return [t for t in range(1, n + 1) if context_at(bits, t, len(c)) == c]


def naive_ctw(bits, depth):
    """CTW_D(x_1:n) from Eq. 14, as an exact Fraction."""

    def ctw(c):
        seq = [bits[t - 1] for t in times_in_context(bits, c, len(bits))]
        xi = kt_prefixes(seq)[-1]
        if len(c) == depth:
            return xi
        return xi / 2 + ctw(c + (0,)) * ctw(c + (1,)) / 2

    return ctw(())


def naive_cts(bits, depth, rate=lambda t: 1.0 / t):
    """CTS_D(x_1:n) from Eq. 16, as a float.

    CTS^c(x_1:n) sums, over every sequence i of "KT" (0) or "children" (1)
    choices, one per symbol seen in c, of w_c(i) times the product over k of
        i_k = 0: xi_KT(first k symbols) / xi_KT(first k-1 symbols)
        i_k = 1: (CTS^0c CTS^1c)(x_1:t) / (CTS^0c CTS^1c)(x_<t),  t = t_c(k).

    w_c is the switch prior of Definition 1 with two models: 1/2 for i_1, then
    each step keeps the previous choice with probability 1 - alpha and flips
    with probability alpha. The paper's update applies alpha_{n+1} after the
    symbol at time n, using the position n in the whole sequence for every
    context. So the step from the (k-1)-th to the k-th symbol of c uses
    alpha = rate(t_c(k-1) + 1).
    """
    memo = {}

    def cts(c, n):
        """CTS^c_{D - len(c)} of the first n symbols x_1:n."""
        if (c, n) in memo:
            return memo[c, n]
        times = times_in_context(bits, c, n)
        kt = kt_prefixes([bits[t - 1] for t in times])
        if len(c) == depth:
            result = float(kt[-1])
        elif not times:
            result = 1.0
        else:
            kt_factor = [float(kt[k + 1] / kt[k]) for k in range(len(times))]
            child_factor = []
            for t in times:
                after = cts(c + (0,), t) * cts(c + (1,), t)
                before = cts(c + (0,), t - 1) * cts(c + (1,), t - 1)
                child_factor.append(after / before)
            alpha = [None] + [rate(times[k - 1] + 1) for k in range(1, len(times))]
            result = 0.0
            for choice in itertools.product((0, 1), repeat=len(times)):
                term = 0.5
                for k in range(1, len(times)):
                    term *= (1 - alpha[k]) if choice[k] == choice[k - 1] else alpha[k]
                for k, picked in enumerate(choice):
                    term *= child_factor[k] if picked else kt_factor[k]
                result += term
        memo[c, n] = result
        return result

    return cts((), len(bits))
