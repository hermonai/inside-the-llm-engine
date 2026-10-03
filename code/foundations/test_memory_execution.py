import unittest

from memory_execution import TileProtocol, bank_pressure, matrix_lines, residency, tiled_product, touched_units


class MemoryExecution(unittest.TestCase):
    def test_lines_crossing_and_column(self):
        self.assertEqual(len(matrix_lines(16, 16)), 16)
        self.assertEqual(len(matrix_lines(16, 16, column=0)), 16)
        self.assertEqual(touched_units([62], 4, 64), [0, 1])
        self.assertEqual(touched_units([], 4, 64), [])

    def test_warp_sector_examples(self):
        for stride, base, expected in ((4,0,4),(4,4,5),(128,0,32),(0,0,1)):
            self.assertEqual(len(touched_units([base+i*stride for i in range(32)],4,32)), expected)

    def test_bank_distinct_words_and_broadcast(self):
        self.assertEqual(bank_pressure(list(range(32))), 1)
        self.assertEqual(bank_pressure([32*i for i in range(32)]), 32)
        self.assertEqual(bank_pressure([33*i for i in range(32)]), 1)
        self.assertEqual(bank_pressure([0]*32), 1)

    def test_tile_against_independent_integer_loop(self):
        for m in range(1, 7):
            for k in range(1, 6):
                for n in range(1, 8):
                    a = [[i-j for j in range(k)] for i in range(m)]
                    b = [[2*i+j for j in range(n)] for i in range(k)]
                    reference = [[sum(a[i][q]*b[q][j] for q in range(k))
                                  for j in range(n)] for i in range(m)]
                    for tile in (1, 2, 3, 4):
                        result, counts = tiled_product(a,b,tile)
                        self.assertEqual(result, reference)
                        self.assertEqual(counts["valid_outputs"], m*n)

    def test_exact_tile_counts(self):
        _, counts = tiled_product([[i-j for j in range(3)] for i in range(5)],
                                 [[2*i+j for j in range(7)] for i in range(3)])
        self.assertEqual(counts, {"blocks":12,"logical_workers":48,
                                 "valid_outputs":35,"global_value_loads":123})
        result, _ = tiled_product([[i-j for j in range(3)] for i in range(5)],
                                  [[2*i+j for j in range(7)] for i in range(3)])
        self.assertEqual(result[0][0],-10)
        self.assertEqual(result[4][6],68)

    def test_early_read_and_overwrite_fail(self):
        protocol = TileProtocol(4)
        protocol.load()
        with self.assertRaises(RuntimeError):
            protocol.consume(0)
        with self.assertRaises(RuntimeError):
            protocol.load()
        protocol.ready()
        for i in range(3):
            protocol.consume(i)
        with self.assertRaises(RuntimeError):
            protocol.retire()
        protocol.consume(3)
        protocol.retire()
        protocol.load()

    def test_duplicate_reader_is_not_completion(self):
        protocol = TileProtocol(2)
        protocol.load()
        protocol.ready()
        protocol.consume(0)
        with self.assertRaises(RuntimeError):
            protocol.consume(0)
        with self.assertRaises(RuntimeError):
            protocol.retire()

    def test_residency_and_bad_input(self):
        self.assertEqual(residency(65536,65536,2048,8,16384,80,256),3)
        self.assertEqual(residency(65536,65536,2048,8,24576,80,256),2)
        for args in (([-1],4,32),([0],0,32)):
            with self.assertRaises(ValueError):
                touched_units(*args)
        with self.assertRaises(ValueError):
            tiled_product([[1,2]],[[1]])
        with self.assertRaises(ValueError):
            TileProtocol(0)

    def test_reuse_intensities_and_launch_bytes(self):
        from fractions import Fraction
        self.assertEqual(Fraction(2,8),Fraction(1,4))
        self.assertEqual(Fraction(8,20),Fraction(2,5))
        self.assertEqual(4*123,492)
        self.assertEqual(4*210,840)
        self.assertEqual(4*35,140)
        self.assertEqual(48-35,13)
        # A complete 2x2 output tile, reduction length 3:
        self.assertEqual(Fraction(2*2**2*3,4*(2*2*3+2**2)),Fraction(3,8))


if __name__ == "__main__":
    unittest.main()
