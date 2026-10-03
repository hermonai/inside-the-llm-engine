import math
import struct
import unittest

from gguf_bytes import Cursor, fixture, inspect, q4_0_values, report, values


class GgufBytes(unittest.TestCase):
    def test_every_byte_and_address(self):
        data = fixture()
        container = inspect(data)
        self.assertEqual((len(data), container.descriptor_end, container.data_origin), (256, 162, 192))
        self.assertEqual(data[:24].hex(), "474755460300000002000000000000000100000000000000")
        self.assertEqual([(t.descriptor_start, t.offset_field) for t in container.tensors],
                         [(57, 101), (109, 154)])
        self.assertEqual([t["absolute_interval"] for t in report(data)["tensors"]],
                         [[192, 224], [224, 242]])
        self.assertEqual(struct.unpack_from("<f", data, 220)[0], 2)

    def test_non_symmetric_projection(self):
        data = fixture()
        container = inspect(data)
        w = values(data, container, container.tensors[0])
        x = [1, 2, 4, 8]
        result = [sum(a * b for a, b in zip(w[p:p + 4], x)) for p in (0, 4)]
        self.assertEqual(result, [49, 19])
        self.assertNotEqual([sum(w[i + j * 2] * x[j] for j in range(4)) for i in range(2)],
                            result, "a wrong orientation must not pass")

    def test_nibble_order_and_nonfinite_scale(self):
        data = fixture()
        actual = q4_0_values(data[224:242])
        expected = [i / 2 for i in range(-8, 8)] + [i / 2 for i in range(7, -9, -1)]
        self.assertEqual(actual, expected)
        self.assertEqual((actual[0], actual[1], actual[16], sum(actual)), (-4, -3.5, 3.5, -8))
        for scale in (math.inf, math.nan):
            with self.assertRaises(ValueError):
                q4_0_values(struct.pack("<e", scale) + bytes(16))
        with self.assertRaises(ValueError):
            q4_0_values(bytes(17))

    def test_pinned_cpu_activation_rounding_prediction(self):
        scale = struct.unpack("<e", struct.pack("<e", 1 / 127))[0]
        factor = 127 * scale
        self.assertEqual(scale, 0.00787353515625)
        self.assertEqual(factor, 0.99993896484375)
        self.assertEqual(-8 * factor, -7.99951171875)

    def test_every_truncation_is_rejected(self):
        data = fixture()
        for length in range(len(data)):
            with self.subTest(length=length), self.assertRaises(ValueError):
                inspect(data[:length])

    def test_typed_metadata_and_utf8_byte_lengths(self):
        data = fixture([("general.name", 8, "café"),
                        ("lab.tokens", 9, ["A", "世"])])
        metadata = inspect(data).metadata
        self.assertEqual(metadata["lab.tokens"], ["A", "世"])
        self.assertEqual(metadata["general.name"], "café")
        self.assertIn(struct.pack("<Q", 3) + "世".encode(), data)
        with self.assertRaises(ValueError):
            inspect(fixture([("general.alignment", 4, 32)]))

    def test_type_namespaces_and_unsupported_values(self):
        data = fixture()
        for position, value in ((49, 2), (97, 4), (4, 4), (53, 24), (53, 0)):
            changed = bytearray(data)
            struct.pack_into("<I", changed, position, value)
            with self.subTest(position=position, value=value), self.assertRaises(ValueError):
                inspect(changed)
        # Nested arrays are valid in the broad spec, unsupported here.
        cursor = Cursor(struct.pack("<IQ", 9, 1))
        with self.assertRaises(ValueError):
            cursor.value(9)

    def test_offsets_overlap_and_shape_limits(self):
        for position, value in ((154, 0), (154, 33), (154, 2**64 - 32),
                                (134, 31), (81, 0), (81, 2**63), (8, 2**63),
                                (24, 2**63)):
            changed = bytearray(fixture())
            struct.pack_into("<Q", changed, position, value)
            with self.subTest(position=position, value=value), self.assertRaises(ValueError):
                inspect(changed)

    def test_padding_and_trailing_bytes(self):
        for position in (162, 191, 242, 255):
            changed = bytearray(fixture())
            changed[position] = 1
            with self.assertRaises(ValueError):
                inspect(changed)
        with self.assertRaises(ValueError):
            inspect(fixture() + b"x")

    def test_container_validity_is_not_numerical_validity(self):
        changed = bytearray(fixture())
        struct.pack_into("<f", changed, 192, math.nan)
        container = inspect(changed)  # structure passes
        with self.assertRaises(ValueError):
            values(changed, container, container.tensors[0])

    def test_duplicate_tensor_names_and_bad_utf8(self):
        for names in (("same", "same"), ("has\0nul", "other"), ("", "other")):
            with self.assertRaises(ValueError):
                inspect(fixture(names=names))
        changed = bytearray(fixture())
        changed[65] = 255
        with self.assertRaises(ValueError):
            inspect(changed)


if __name__ == "__main__":
    unittest.main()
