"""Krichevsky-Trofimov (KT) estimator for a memoryless binary source (paper section 2.4.2)."""
from __future__ import annotations

from math import exp, log

LN2 = log(2)


def kt_prob(bit: int, zeros: int, ones: int) -> float:
    """Eq. 7: the probability of `bit` after seeing `zeros` 0s and `ones` 1s."""
    seen = ones if bit else zeros
    return (seen + 0.5) / (zeros + ones + 1)


class KTEstimator:
    """Sequential KT predictor.

    It is the depth-0 case of both CTW and CTS, so it serves as their baseline.
    `log_joint` is ln of the KT probability of everything seen so far (Eq. 6),
    built up with the chain rule from the conditionals of Eq. 7.
    """

    def __init__(self):
        self.zeros = 0
        self.ones = 0
        self.log_joint = 0.0

    @property
    def code_length(self) -> float:
        """-log2 of the joint probability, in bits."""
        return -self.log_joint / LN2

    def log_prob(self, bit: int) -> float:
        """ln P(next bit = `bit` | history)."""
        return log(kt_prob(bit, self.zeros, self.ones))

    def predict(self) -> float:
        """P(next bit = 1 | history)."""
        return exp(self.log_prob(1))

    def update(self, bit: int) -> float:
        """Observe `bit`. Returns ln P(bit | history), as it was before the update."""
        if bit not in (0, 1):
            raise ValueError("bit must be 0 or 1")
        log_p = self.log_prob(bit)
        self.log_joint += log_p
        if bit:
            self.ones += 1
        else:
            self.zeros += 1
        return log_p
