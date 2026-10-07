"""Bit-level helpers shared by the arithmetic coder, the codec and the benchmark."""
from __future__ import annotations

from typing import Iterable, Iterator


def iter_bits(data: bytes) -> Iterator[int]:
    """The bits of `data`, each byte most significant bit first."""
    for byte in data:
        for shift in range(7, -1, -1):
            yield (byte >> shift) & 1


def pack_bits(bits: Iterable[int]) -> bytes:
    """Inverse of `iter_bits`. A final partial byte is padded with zero bits."""
    out = bytearray()
    byte = count = 0
    for bit in bits:
        byte = (byte << 1) | bit
        count += 1
        if count == 8:
            out.append(byte)
            byte = count = 0
    if count:
        out.append(byte << (8 - count))
    return bytes(out)
