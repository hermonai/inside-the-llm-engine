"""A bounded GGUF v3 teaching reader, fixture writer and scalar decoder.

Supported subset: little endian; UINT32/STRING/ARRAY-of-STRING metadata;
positive 1-4D F32 or Q4_0 tensors; at most 1 MiB. This is not a model loader.
Default command only operates in memory. --write creates a NEW fixture file.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import struct

LIMIT = 1 << 20
TYPE_BLOCKS = {0: (1, 4), 2: (32, 18)}


def align_up(position, alignment):
    return position + (-position % alignment)


class Cursor:
    def __init__(self, data):
        if len(data) > LIMIT:
            raise ValueError("teaching file-size limit exceeded")
        self.data, self.position = data, 0

    def take(self, count):
        if count < 0 or count > len(self.data) - self.position:
            raise ValueError("truncated field or payload")
        start = self.position
        self.position += count
        return self.data[start:self.position]

    def number(self, code):
        return struct.unpack("<" + code, self.take(struct.calcsize("<" + code)))[0]

    def string(self, maximum=65535):
        length = self.number("Q")
        if length > maximum:
            raise ValueError("string length exceeds teaching policy")
        try:
            return self.take(length).decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ValueError("invalid UTF-8") from error

    def value(self, kind):
        if kind == 4:
            return self.number("I")
        if kind == 8:
            return self.string()
        if kind == 9:
            element_kind, count = self.number("I"), self.number("Q")
            if element_kind != 8 or count > 1024:
                raise ValueError("only bounded arrays of strings are supported")
            return [self.string() for _ in range(count)]
        raise ValueError("metadata type unsupported by this teaching reader")


@dataclass(frozen=True)
class Tensor:
    name: str
    dimensions: tuple
    kind: int
    offset: int
    nbytes: int
    descriptor_start: int
    offset_field: int


@dataclass(frozen=True)
class Container:
    metadata: dict
    tensors: tuple
    descriptor_end: int
    data_origin: int
    alignment: int


def inspect(data):
    cursor = Cursor(data)
    if cursor.take(4) != b"GGUF" or cursor.number("I") != 3:
        raise ValueError("expected little-endian GGUF version 3")
    count, metadata_count = cursor.number("Q"), cursor.number("Q")
    if not 1 <= count <= 1024 or metadata_count > 1024:
        raise ValueError("tensor/metadata count outside teaching policy")
    metadata, kinds = {}, {}
    for _ in range(metadata_count):
        key = cursor.string()
        if not re.fullmatch(r"[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+", key):
            raise ValueError("invalid metadata key in teaching subset")
        if key in metadata:
            raise ValueError("duplicate metadata key")
        kind = cursor.number("I")
        kinds[key] = kind
        metadata[key] = cursor.value(kind)
    alignment = metadata.get("general.alignment", 32)
    if "general.alignment" in kinds and kinds["general.alignment"] != 4:
        raise ValueError("alignment must be UINT32")
    if not 8 <= alignment <= 4096 or alignment & (alignment - 1):
        raise ValueError("teaching alignment must be a power of two in [8,4096]")

    tensors, names = [], set()
    for _ in range(count):
        start = cursor.position
        name = cursor.string(64)
        if not name or "\0" in name or name in names:
            raise ValueError("empty, NUL-containing or duplicate tensor name")
        names.add(name)
        rank = cursor.number("I")
        if not 1 <= rank <= 4:
            raise ValueError("unsupported tensor rank")
        dimensions = tuple(cursor.number("Q") for _ in range(rank))
        elements = 1
        for extent in dimensions:
            if extent == 0 or extent > LIMIT // elements:
                raise ValueError("tensor dimensions exceed teaching policy")
            elements *= extent
        kind = cursor.number("I")
        if kind not in TYPE_BLOCKS:
            raise ValueError("tensor type unsupported by this teaching reader")
        block_elements, block_bytes = TYPE_BLOCKS[kind]
        if dimensions[0] % block_elements:
            raise ValueError("fastest dimension must contain whole blocks")
        size = elements // block_elements * block_bytes
        offset_field = cursor.position
        offset = cursor.number("Q")
        if offset % alignment:
            raise ValueError("misaligned tensor offset")
        tensors.append(Tensor(name, dimensions, kind, offset, size, start, offset_field))

    end = cursor.position
    origin = align_up(end, alignment)
    if origin > len(data) or any(data[end:origin]):
        raise ValueError("missing or nonzero directory padding")
    occupied_end = 0
    for tensor in sorted(tensors, key=lambda t: t.offset):
        if tensor.offset < occupied_end:
            raise ValueError("overlapping tensor allocations")
        if tensor.offset > len(data) - origin or tensor.nbytes > len(data) - origin - tensor.offset:
            raise ValueError("tensor payload is outside the file")
        if any(data[origin + occupied_end:origin + tensor.offset]):
            raise ValueError("nonzero inter-tensor padding")
        payload_end = tensor.offset + tensor.nbytes
        occupied_end = align_up(payload_end, alignment)
        if occupied_end > len(data) - origin or any(data[origin + payload_end:origin + occupied_end]):
            raise ValueError("missing or nonzero tensor padding")
    if origin + occupied_end != len(data):
        raise ValueError("trailing bytes outside the teaching container")
    return Container(metadata, tuple(tensors), end, origin, alignment)


def q4_0_values(block):
    if len(block) != 18:
        raise ValueError("Q4_0 block needs exactly 18 bytes")
    scale = struct.unpack("<e", block[:2])[0]
    if not math.isfinite(scale):
        raise ValueError("nonfinite scale rejected by numerical policy")
    low = [(byte & 15) - 8 for byte in block[2:]]
    high = [(byte >> 4) - 8 for byte in block[2:]]
    return [scale * code for code in low + high]


def values(data, container, tensor):
    start = container.data_origin + tensor.offset
    raw = data[start:start + tensor.nbytes]
    if tensor.kind == 0:
        result = list(struct.unpack("<" + "f" * (len(raw) // 4), raw))
    else:
        result = [x for p in range(0, len(raw), 18) for x in q4_0_values(raw[p:p + 18])]
    if not all(math.isfinite(x) for x in result):
        raise ValueError("nonfinite weight rejected by numerical policy")
    return result


def _string(text):
    encoded = text.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def fixture(extra_metadata=(), names=("dense.weight", "packed.weight")):
    """Return a canonical container, not a trained model or quantizer."""
    metadata = [("general.alignment", 4, 32), *extra_metadata]
    output = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 2, len(metadata)))
    for key, kind, value in metadata:
        output += _string(key) + struct.pack("<I", kind)
        if kind == 4:
            output += struct.pack("<I", value)
        elif kind == 8:
            output += _string(value)
        elif kind == 9:
            output += struct.pack("<IQ", 8, len(value)) + b"".join(_string(v) for v in value)
        else:
            raise ValueError("fixture metadata kind unsupported")
    for name, dimensions, kind, offset in (
            (names[0], (4, 2), 0, 0), (names[1], (32, 1), 2, 32)):
        output += _string(name) + struct.pack("<I", 2)
        output += struct.pack("<QQIQ", *dimensions, kind, offset)
    output += bytes(-len(output) % 32)
    output += struct.pack("<8f", 1, 2, 3, 4, -1, 0, 1, 2)
    output += struct.pack("<e", .5) + bytes(i | ((15 - i) << 4) for i in range(16))
    output += bytes(-len(output) % 32)
    return bytes(output)


def report(data):
    container = inspect(data)
    return {"sha256": hashlib.sha256(data).hexdigest(), "file_bytes": len(data),
            "descriptor_end": container.descriptor_end, "data_origin": container.data_origin,
            "tensors": [{"name": t.name, "ne": t.dimensions, "type": t.kind,
                         "relative_offset": t.offset, "payload_bytes": t.nbytes,
                         "absolute_interval": [container.data_origin + t.offset,
                                               container.data_origin + t.offset + t.nbytes],
                         "values": values(data, container, t)} for t in container.tensors]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", type=Path, help="write a new fixture; never overwrite")
    group.add_argument("--inspect", type=Path, help="inspect a file within the limited subset")
    args = parser.parse_args()
    data = fixture()
    if args.inspect:
        with args.inspect.open("rb") as stream:
            data = stream.read(LIMIT + 1)
    if args.write:
        with args.write.open("xb") as stream:
            stream.write(data)
    print(json.dumps(report(data), indent=2))


if __name__ == "__main__":
    main()
