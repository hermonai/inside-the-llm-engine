"""Address sets and cooperative-tile contracts; no hardware timings or scheduling."""
import json


def touched_units(addresses, width, unit):
    if width <= 0 or unit <= 0 or any(a < 0 for a in addresses):
        raise ValueError("nonnegative addresses and positive widths required")
    return sorted({bucket for a in addresses
                   for bucket in range(a // unit, (a + width - 1) // unit + 1)})


def matrix_lines(rows, columns, line=64, column=None):
    if rows < 1 or columns < 1 or (column is not None and not 0 <= column < columns):
        raise ValueError("invalid matrix or column")
    addresses = ([4 * (i * columns + j) for i in range(rows) for j in range(columns)]
                 if column is None else [4 * (i * columns + column) for i in range(rows)])
    return touched_units(addresses, 4, line)


def bank_pressure(words, banks=32):
    if banks < 1 or any(word < 0 for word in words):
        raise ValueError("invalid bank layout")
    unique = [set() for _ in range(banks)]
    for word in words:
        unique[word % banks].add(word)
    return max(map(len, unique), default=0)


class TileProtocol:
    """Explicit producer/consumer stages; this does not emulate CUDA barriers."""
    def __init__(self, readers):
        if type(readers) is not int or readers < 1:
            raise ValueError("positive integer consumer count required")
        self.readers = readers
        self.state = "empty"
        self.seen = set()

    def load(self):
        if self.state not in ("empty", "retired"):
            raise RuntimeError("overwrite before all readers finish")
        self.state = "loading"
        self.seen.clear()

    def ready(self):
        if self.state != "loading":
            raise RuntimeError("publish requires completed loading stage")
        self.state = "ready"

    def consume(self, reader):
        if self.state != "ready" or not 0 <= reader < self.readers or reader in self.seen:
            raise RuntimeError("invalid, early or duplicate consumer")
        self.seen.add(reader)

    def retire(self):
        if len(self.seen) != self.readers or self.state != "ready":
            raise RuntimeError("reuse requires every consumer to finish")
        self.state = "retired"


def tiled_product(a, b, tile=2):
    """Serial cooperative-tile model. All logical workers participate at edges."""
    if not a or not b or not b[0] or type(tile) is not int or tile < 1:
        raise ValueError("positive shapes and tile required")
    m, k, n = len(a), len(b), len(b[0])
    if any(len(row) != k for row in a) or any(len(row) != n for row in b):
        raise ValueError("ragged or mismatched matrices")
    result = [[0 for _ in range(n)] for _ in range(m)]
    loads, blocks, valid = 0, 0, 0
    for row0 in range(0, m, tile):
        for col0 in range(0, n, tile):
            blocks += 1
            protocol = TileProtocol(tile * tile)
            accumulators = [[0] * tile for _ in range(tile)]
            for k0 in range(0, k, tile):
                protocol.load()
                left, right = [[0]*tile for _ in range(tile)], [[0]*tile for _ in range(tile)]
                for i in range(tile):
                    for j in range(tile):
                        if row0+i < m and k0+j < k:
                            left[i][j] = a[row0+i][k0+j]
                            loads += 1
                        if k0+i < k and col0+j < n:
                            right[i][j] = b[k0+i][col0+j]
                            loads += 1
                protocol.ready()
                for i in range(tile):
                    for j in range(tile):
                        for q in range(tile):
                            accumulators[i][j] += left[i][q] * right[q][j]
                        protocol.consume(i * tile + j)
                protocol.retire()
            for i in range(tile):
                for j in range(tile):
                    if row0+i < m and col0+j < n:
                        result[row0+i][col0+j] = accumulators[i][j]
                        valid += 1
    return result, {"blocks": blocks, "logical_workers": blocks*tile*tile,
                    "valid_outputs": valid, "global_value_loads": loads}


def residency(shared, registers, threads, block_limit, per_block_shared, per_thread_registers,
              block_threads):
    """Illustrative integer ceilings, without architecture allocation granularity."""
    if min(shared, registers, threads, block_limit, per_block_shared,
           per_thread_registers, block_threads) < 1:
        raise ValueError("positive capacities required")
    return min(block_limit, shared // per_block_shared,
               registers // (per_thread_registers * block_threads), threads // block_threads)


def demonstrate():
    a = [[i-j for j in range(3)] for i in range(5)]
    b = [[2*i+j for j in range(7)] for i in range(3)]
    _, counts = tiled_product(a, b)
    return {"f32_16x16_row_scan_lines": len(matrix_lines(16, 16)),
            "f32_16x16_one_column_lines": len(matrix_lines(16, 16, column=0)),
            "aligned_warp_sectors": len(touched_units([4*i for i in range(32)], 4, 32)),
            "offset_warp_sectors": len(touched_units([4+4*i for i in range(32)], 4, 32)),
            "strided_warp_sectors": len(touched_units([128*i for i in range(32)], 4, 32)),
            "shared_column_conflict": bank_pressure([32*i for i in range(32)]),
            "shared_padded_column_conflict": bank_pressure([33*i for i in range(32)]),
            "shared_broadcast_conflict": bank_pressure([0]*32),
            "tile_5x3_by_3x7": counts,
            "storage_only_blocks": 64//16,
            "all_limits_blocks": residency(65536,65536,2048,8,16384,80,256)}


if __name__ == "__main__":
    print(json.dumps(demonstrate(), indent=2))
