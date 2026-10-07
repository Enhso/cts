"""A small integer binary arithmetic coder (Witten, Neal and Cleary, 1987).

The coder keeps an interval [low, high] of PRECISION-bit integers. Coding a bit
splits the interval in proportion to the model's probability and keeps the
sub-interval of the bit that occurred; leading bits that both ends agree on are
shifted out as output. When the interval straddles the midpoint but is short, a
"pending" counter postpones the decision (the carry-less trick of the paper),
so no output bit is ever revised.

Probabilities are integers: `p1` is P(bit = 1) in units of 1 / PROB_ONE, kept in
[1, PROB_ONE - 1] so that both bits always get a non-empty sub-interval.
"""
from __future__ import annotations

from src.utils.io import pack_bits

PRECISION = 32
TOP = (1 << PRECISION) - 1
HALF = 1 << (PRECISION - 1)
QUARTER = 1 << (PRECISION - 2)

PROB_BITS = 16
PROB_ONE = 1 << PROB_BITS


def quantize(p1: float) -> int:
    """Turn P(bit = 1) into the integer the coder uses, clamped away from 0 and 1."""
    return min(max(round(p1 * PROB_ONE), 1), PROB_ONE - 1)


def _split(low: int, high: int, p1: int) -> int:
    """First value of the "bit = 0" part: bit 1 gets [low, split), bit 0 gets [split, high]."""
    return low + (((high - low + 1) * p1) >> PROB_BITS)


class Encoder:
    def __init__(self):
        self.low = 0
        self.high = TOP
        self.pending = 0  # undecided output bits (a run of the form 0111...1 or 1000...0)
        self.bits: list[int] = []

    def _emit(self, bit: int) -> None:
        self.bits.append(bit)
        self.bits.extend([1 - bit] * self.pending)
        self.pending = 0

    def encode(self, bit: int, p1: int) -> None:
        split = _split(self.low, self.high, p1)
        if bit:
            self.high = split - 1
        else:
            self.low = split
        while True:
            if self.high < HALF:
                self._emit(0)
            elif self.low >= HALF:
                self._emit(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < HALF + QUARTER:
                self.pending += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                return
            self.low <<= 1
            self.high = (self.high << 1) | 1

    def finish(self) -> bytes:
        """Flush enough bits to pin down the final interval and return the code."""
        self.pending += 1
        self._emit(0 if self.low < QUARTER else 1)
        return pack_bits(self.bits)


class Decoder:
    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0  # next bit to read; reads past the end give 0s
        self.low = 0
        self.high = TOP
        self.value = 0
        for _ in range(PRECISION):
            self.value = (self.value << 1) | self._read_bit()

    def _read_bit(self) -> int:
        index = self._pos >> 3
        byte = self._data[index] if index < len(self._data) else 0
        bit = (byte >> (7 - (self._pos & 7))) & 1
        self._pos += 1
        return bit

    def decode(self, p1: int) -> int:
        split = _split(self.low, self.high, p1)
        if self.value < split:
            bit = 1
            self.high = split - 1
        else:
            bit = 0
            self.low = split
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.value -= HALF
            elif self.low >= QUARTER and self.high < HALF + QUARTER:
                self.low -= QUARTER
                self.high -= QUARTER
                self.value -= QUARTER
            else:
                return bit
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.value = (self.value << 1) | self._read_bit()
