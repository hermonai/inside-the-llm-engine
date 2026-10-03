"""Exact illustrative accounting. Not a hardware or inference benchmark."""
from dataclasses import dataclass
from fractions import Fraction
import json


UNITS = {
    "byte": (1, 0, 0), "FLOP": (0, 1, 0), "s": (0, 0, 1),
    "byte/s": (1, 0, -1), "FLOP/s": (0, 1, -1),
    "FLOP/byte": (-1, 1, 0),
}


@dataclass(frozen=True)
class Quantity:
    value: Fraction
    dimensions: tuple

    @classmethod
    def of(cls, value, unit):
        return cls(Fraction(value), UNITS[unit])

    def __add__(self, other):
        if self.dimensions != other.dimensions:
            raise ValueError("cannot add unlike dimensions")
        return Quantity(self.value + other.value, self.dimensions)

    def __truediv__(self, other):
        return Quantity(self.value / other.value,
                        tuple(a-b for a, b in zip(self.dimensions, other.dimensions)))

    def in_unit(self, unit):
        if self.dimensions != UNITS[unit]:
            raise ValueError("wrong result dimension")
        return self.value


def integer(value, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError("expected an integer within the declared range")
    return value


def packed_bytes(elements, group, code_bytes, metadata_bytes):
    """An illustrative full-group format; no silently padded trailing group."""
    for value in (elements, group, code_bytes):
        integer(value)
    integer(metadata_bytes, 0)
    if elements % group:
        raise ValueError("format requires complete groups")
    return (elements // group) * (code_bytes + metadata_bytes)


def projection(rows, inputs, outputs, weight_bytes, activation_bytes=2):
    """One weight fetch, each input/output once, at one stated boundary."""
    for value in (rows, inputs, outputs, weight_bytes, activation_bytes):
        integer(value)
    incoming = rows * inputs * activation_bytes
    outgoing = rows * outputs * activation_bytes
    traffic = weight_bytes + incoming + outgoing
    flops = 2 * rows * inputs * outputs  # multiply-add counting convention
    return {"weights": inputs*outputs, "input_bytes": incoming,
            "output_bytes": outgoing, "traffic_bytes": traffic,
            "flops": flops, "intensity": Fraction(flops, traffic)}


def cache_bytes(lengths, bytes_per_token, block_tokens, table_entry_bytes=4):
    """Private, unshared blocks; empty requests reserve no blocks."""
    for value in (bytes_per_token, block_tokens, table_entry_bytes):
        integer(value)
    lengths = tuple(integer(n, 0) for n in lengths)
    blocks = sum((n + block_tokens - 1)//block_tokens for n in lengths)
    logical = sum(lengths)*bytes_per_token
    reserved = blocks*block_tokens*bytes_per_token
    return {"logical_bytes": logical, "reserved_bytes": reserved,
            "tail_bytes": reserved-logical, "blocks": blocks,
            "table_bytes": blocks*table_entry_bytes}


def bounds(traffic, work, bandwidth, compute):
    """A conditional resource bound; caller must establish unavoidable work."""
    d = traffic.in_unit("byte")
    f = work.in_unit("FLOP")
    beta = bandwidth.in_unit("byte/s")
    pi = compute.in_unit("FLOP/s")
    if d < 0 or f < 0 or beta <= 0 or pi <= 0:
        raise ValueError("work must be nonnegative and ceilings positive")
    memory_s = (traffic / bandwidth).in_unit("s")
    compute_s = (work / compute).in_unit("s")
    return {"memory_s": memory_s, "compute_s": compute_s,
            "ideal_overlap_s": max(memory_s, compute_s),
            "serial_resource_s": memory_s+compute_s}


def main():
    matrix = projection(3, 16, 8, packed_bytes(128, 16, 8, 2))
    cache = cache_bytes([1, 5, 9], 64, 4)
    times = bounds(Quantity.of(2_000_000_000, "byte"),
                   Quantity.of(8_000_000_000, "FLOP"),
                   Quantity.of(100_000_000_000, "byte/s"),
                   Quantity.of(1_000_000_000_000, "FLOP/s"))
    print(json.dumps({"evidence_kind": "derived from illustrative inputs",
                      "projection": matrix, "private_cache": cache,
                      "conditional_times": times}, default=str, indent=2))


if __name__ == "__main__":
    main()
