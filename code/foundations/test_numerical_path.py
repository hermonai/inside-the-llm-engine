from fractions import Fraction
import math
import struct
import unittest

from numerical_path import BF16, F16, F32, accumulated, nearest_even, power2, quantized


class NumericalPath(unittest.TestCase):
    def test_all_half_decodings_against_struct(self):
        for bits in range(65536):
            reference, = struct.unpack("<e", struct.pack("<H", bits))
            value = F16.decode(bits)
            if math.isnan(reference):
                self.assertEqual(value, "nan")
            elif math.isinf(reference):
                self.assertEqual(value, "-inf" if reference < 0 else "+inf")
            else:
                self.assertEqual(value, Fraction(reference))
                if value:
                    self.assertEqual(F16.encode(value), bits)

    def test_half_midpoints_against_struct(self):
        # Dyadic half midpoints are exact in the Python float supplied to struct.
        for bits in range(0, 0x7BFF, 31):
            lo, hi = F16.decode(bits), F16.decode(bits + 1)
            midpoint = (lo + hi) / 2
            for value in (midpoint, midpoint - (hi-lo)/8, midpoint + (hi-lo)/8):
                for signed in (value, -value):
                    reference, = struct.unpack("<H", struct.pack("<e", float(signed)))
                    self.assertEqual(F16.encode(signed), reference)

    def test_bfloat_midpoints(self):
        for bits in range(1, 0x7F7F, 37):
            midpoint = (BF16.decode(bits) + BF16.decode(bits + 1)) / 2
            self.assertEqual(BF16.encode(midpoint), bits if bits % 2 == 0 else bits + 1)

    def test_special_fields_and_signed_zero(self):
        self.assertEqual(F16.fields(0x8000), (1, 0, 0))
        self.assertEqual(F16.decode(0x8000), 0)
        self.assertEqual(F16.decode(0x7C00), "+inf")
        self.assertEqual(F16.decode(0xFC00), "-inf")
        self.assertEqual(F16.decode(0x7E00), "nan")
        self.assertEqual(F16.encode(65520), 0x7C00)
        self.assertEqual(F16.encode(Fraction(65520) - Fraction(1, 4)), 0x7BFF)
        for invalid in (-1, 65536, True):
            with self.assertRaises(ValueError):
                F16.fields(invalid)

    def test_subnormal_tie_and_boundary(self):
        self.assertEqual(F16.decode(1), power2(-24))
        self.assertEqual(F16.encode(power2(-25)), 0)
        self.assertEqual(F16.encode(3 * power2(-25)), 2)
        self.assertEqual(F16.encode(power2(-14)), 0x400)

    def test_spacing_and_encoding(self):
        self.assertEqual(F16.encode(Fraction(3, 2)), 0x3E00)
        self.assertEqual(BF16.encode(Fraction(3, 2)), 0x3FC0)
        self.assertEqual(F16.decode(0x3C01) - 1, power2(-10))
        self.assertEqual(BF16.decode(0x3F81) - 1, power2(-7))
        self.assertEqual(F32.decode(0x3F800001) - 1, power2(-23))

    def test_accumulation_and_order(self):
        self.assertEqual(accumulated([1] * 4096, F16), 2048)
        self.assertEqual(accumulated([1] * 4096, F32), 4096)
        self.assertEqual(accumulated([2**24, 1, -2**24], F32), 0)
        self.assertEqual(accumulated([2**24, -2**24, 1], F32), 1)

    def test_exact_fused_example(self):
        epsilon = power2(-13)
        a, b = 1 + epsilon, 1 - epsilon
        self.assertEqual(F32.round(a), a)
        self.assertEqual(F32.round(b), b)
        self.assertEqual(F32.round(F32.round(a*b) - 1), 0)
        self.assertEqual(F32.round(a*b - 1), -power2(-26))

    def test_quantization_ties_clipping_and_outlier(self):
        self.assertEqual(nearest_even(Fraction(5, 2)), 2)
        codes, values = quantized([Fraction(5, 4), Fraction(-5, 4), 10], Fraction(1, 2))
        self.assertEqual(codes, [2, -2, 7])
        self.assertEqual(values, [1, -1, Fraction(7, 2)])
        codes, _ = quantized([Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), 28], 4)
        self.assertEqual(codes, [0, 0, 0, 7])
        with self.assertRaises(ValueError):
            quantized([1], 0)

    def test_projection_error_and_group_scale(self):
        weights = [Fraction(-11,10), Fraction(1,5), Fraction(8,5)]
        inputs = [1,2,-1]
        _, reconstructed = quantized(weights, Fraction(1,2))
        reference = sum(w*x for w,x in zip(weights,inputs))
        approximate = sum(w*x for w,x in zip(reconstructed,inputs))
        bound = sum(abs(w-r)*abs(x) for w,r,x in zip(weights,reconstructed,inputs))
        self.assertEqual(reference, Fraction(-23,10))
        self.assertEqual(approximate, Fraction(-5,2))
        self.assertEqual(bound, Fraction(3,5))
        self.assertLessEqual(abs(reference-approximate),bound)
        self.assertEqual(quantized([Fraction(1,4),Fraction(1,2),Fraction(3,4)],
                                   Fraction(3,28))[0], [2,5,7])


if __name__ == "__main__":
    unittest.main()
