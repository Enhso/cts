"""Byte-level compression: bytes are split into bits, most significant first, and
each bit is coded by an arithmetic coder driven by a CTS or CTW model.

This is the paper's base setup (section 4): contexts are the previous `depth`
bits in the bit stream, whether or not they cross a byte boundary, and no
enhancements are used. The paper's CTW48 and CTS48 are depth = 48.

File format: a 14-byte header (magic, model id, depth, original length in bytes)
followed by the arithmetic-coded bits.
"""
from __future__ import annotations

import struct

from src.utils.io import iter_bits

from .arithmetic_coder import Decoder, Encoder, quantize
from .cts_model import CTSModel, CTWModel

MAGIC = b"CTS1"
HEADER = struct.Struct(">4sBBQ")  # magic, model id, depth, length in bytes
MODELS = {"cts": (0, CTSModel), "ctw": (1, CTWModel)}
MODEL_BY_ID = {model_id: cls for model_id, cls in MODELS.values()}
MAX_DEPTH = 255


def _check(model: str, depth: int) -> None:
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {sorted(MODELS)}")
    if not 0 <= depth <= MAX_DEPTH:
        raise ValueError(f"depth must be between 0 and {MAX_DEPTH}")


def code_length(data: bytes, model: str = "cts", depth: int = 48) -> float:
    """Ideal code length of `data` in bits: -log2 of the model's probability of it."""
    _check(model, depth)
    predictor = MODELS[model][1](depth)
    for bit in iter_bits(data):
        predictor.update(bit)
    return predictor.code_length


def compress(data: bytes, model: str = "cts", depth: int = 48) -> bytes:
    _check(model, depth)
    model_id, model_cls = MODELS[model]
    predictor = model_cls(depth)
    encoder = Encoder()
    for bit in iter_bits(data):
        encoder.encode(bit, quantize(predictor.predict()))
        predictor.update(bit)
    return HEADER.pack(MAGIC, model_id, depth, len(data)) + encoder.finish()


def decompress(blob: bytes) -> bytes:
    if len(blob) < HEADER.size:
        raise ValueError("input is too short to be a compressed file")
    magic, model_id, depth, length = HEADER.unpack_from(blob)
    if magic != MAGIC or model_id not in MODEL_BY_ID:
        raise ValueError("input is not a file produced by this compressor")
    predictor = MODEL_BY_ID[model_id](depth)
    decoder = Decoder(blob[HEADER.size:])
    out = bytearray()
    for _ in range(length):
        byte = 0
        for _ in range(8):
            bit = decoder.decode(quantize(predictor.predict()))
            predictor.update(bit)
            byte = (byte << 1) | bit
        out.append(byte)
    return bytes(out)
