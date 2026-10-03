"""Exact finite-input binary rounding; teaching model, not a hardware emulator."""
from dataclasses import dataclass
from fractions import Fraction
import json


def power2(exponent):
    return Fraction(2 ** exponent) if exponent >= 0 else Fraction(1, 2 ** -exponent)


def nearest_even(value):
    """Nonnegative rational to integer, with an explicit halfway rule."""
    whole, remainder = divmod(value.numerator, value.denominator)
    twice = 2 * remainder
    return whole + int(twice > value.denominator or
                       (twice == value.denominator and whole % 2 == 1))


@dataclass(frozen=True)
class BinaryFormat:
    exponent_bits: int
    fraction_bits: int

    @property
    def bias(self):
        return (1 << (self.exponent_bits - 1)) - 1

    @property
    def width(self):
        return 1 + self.exponent_bits + self.fraction_bits

    def fields(self, bits):
        if type(bits) is not int or not 0 <= bits < 1 << self.width:
            raise ValueError("encoding outside format")
        return (bits >> (self.width - 1),
                (bits >> self.fraction_bits) & ((1 << self.exponent_bits) - 1),
                bits & ((1 << self.fraction_bits) - 1))

    def decode(self, bits):
        """Finite encodings return exact Fraction; specials retain their class."""
        sign, exponent, fraction = self.fields(bits)
        if exponent == (1 << self.exponent_bits) - 1:
            return "nan" if fraction else ("-inf" if sign else "+inf")
        significand = fraction if exponent == 0 else (1 << self.fraction_bits) + fraction
        unbiased = 1 - self.bias if exponent == 0 else exponent - self.bias
        return (-1 if sign else 1) * significand * power2(unbiased - self.fraction_bits)

    def encode(self, value):
        """Round a finite rational once. Rational zero has no negative sign."""
        value = Fraction(value)
        sign = int(value < 0) << (self.width - 1)
        magnitude = abs(value)
        if not magnitude:
            return 0
        exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
        if magnitude < power2(exponent):
            exponent -= 1
        exponent = max(exponent, 1 - self.bias)
        units = nearest_even(magnitude / power2(exponent - self.fraction_bits))
        if units == 1 << (self.fraction_bits + 1):
            units //= 2
            exponent += 1
        if exponent > self.bias:
            return sign | (((1 << self.exponent_bits) - 1) << self.fraction_bits)
        field = 0 if units < 1 << self.fraction_bits else exponent + self.bias
        return sign | (field << self.fraction_bits) | (units & ((1 << self.fraction_bits) - 1))

    def round(self, value):
        result = self.decode(self.encode(value))
        if not isinstance(result, Fraction):
            raise OverflowError(result)
        return result


F16 = BinaryFormat(5, 10)
BF16 = BinaryFormat(8, 7)
F32 = BinaryFormat(8, 23)


def accumulated(values, fmt):
    result = Fraction(0)
    for value in values:
        result = fmt.round(result + value)
    return result


def quantized(values, scale, maximum=7):
    scale = Fraction(scale)
    if scale <= 0 or maximum < 1:
        raise ValueError("positive scale and code limit required")
    codes = []
    for value in values:
        quotient = Fraction(value) / scale
        code = nearest_even(abs(quotient)) * (-1 if quotient < 0 else 1)
        codes.append(max(-maximum, min(maximum, code)))
    return codes, [scale * code for code in codes]


def demonstrate():
    epsilon = power2(-13)
    product = (1 + epsilon) * (1 - epsilon)
    values = [Fraction(1, 4), Fraction(1, 2), Fraction(3, 4), Fraction(28)]
    codes, reconstructed = quantized(values, Fraction(4))
    return {
        "f16_1_5_hex": hex(F16.encode(Fraction(3, 2))),
        "bf16_1_5_hex": hex(BF16.encode(Fraction(3, 2))),
        "f16_min_subnormal": str(F16.decode(1)),
        "sum_4096_ones_f16": str(accumulated([1] * 4096, F16)),
        "sum_4096_ones_f32": str(accumulated([1] * 4096, F32)),
        "serial_cancellation": str(accumulated([2**24, 1, -2**24], F32)),
        "regrouped_cancellation": str(accumulated([2**24, -2**24, 1], F32)),
        "separate_multiply_add": str(F32.round(F32.round(product) - 1)),
        "fused_multiply_add": str(F32.round(product - 1)),
        "outlier_codes": codes,
        "outlier_reconstruction": [str(x) for x in reconstructed],
    }


if __name__ == "__main__":
    print(json.dumps(demonstrate(), indent=2))
