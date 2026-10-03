from fractions import Fraction
import unittest
from performance_contract import Quantity, packed_bytes, projection, cache_bytes, bounds


class PerformanceContract(unittest.TestCase):
    def test_units_cancel_and_reject_plausible_wrong_division(self):
        traffic = Quantity.of(2_000_000_000, "byte")
        beta = Quantity.of(100_000_000_000, "byte/s")
        self.assertEqual((traffic/beta).in_unit("s"), Fraction(1,50))
        with self.assertRaises(ValueError):
            (Quantity.of(8_000_000_000, "FLOP")/beta).in_unit("s")
        with self.assertRaises(ValueError):
            traffic + Quantity.of(8, "FLOP")

    def test_projection_count_against_loop_enumeration(self):
        for rows in range(1,5):
            for inputs in range(1,6):
                for outputs in range(1,5):
                    counted = sum(2 for _ in range(rows)
                                  for _ in range(inputs) for _ in range(outputs))
                    self.assertEqual(projection(rows, inputs, outputs, 2*inputs*outputs)["flops"], counted)
        self.assertEqual(projection(3,16,8,80), {
            "weights":128,"input_bytes":96,"output_bytes":48,
            "traffic_bytes":224,"flops":768,"intensity":Fraction(24,7)})

    def test_metadata_and_batch_do_not_duplicate_parameters(self):
        self.assertEqual(packed_bytes(128,16,8,2),80)
        self.assertEqual(Fraction(80,128),Fraction(5,8))
        self.assertEqual(projection(1,16,8,80)["weights"],
                         projection(3,16,8,80)["weights"])
        with self.assertRaises(ValueError):
            packed_bytes(127,16,8,2)

    def test_private_reservation_and_tables(self):
        self.assertEqual(cache_bytes([1,5,9],64,4),{
            "logical_bytes":960,"reserved_bytes":1536,
            "tail_bytes":576,"blocks":6,"table_bytes":24})
        self.assertEqual(cache_bytes([0,4,8],64,4)["tail_bytes"],0)
        # Enumerate reserved positions independently of ceiling division.
        for n in range(33):
            slots = list(range(n))
            while len(slots) % 4:
                slots.append(None)
            self.assertEqual(cache_bytes([n],64,4)["reserved_bytes"],len(slots)*64)

    def test_bounds_exact_and_invalid_dimensions(self):
        args = [Quantity.of(v,u) for v,u in (
            (2_000_000_000,"byte"),(8_000_000_000,"FLOP"),
            (100_000_000_000,"byte/s"),(1_000_000_000_000,"FLOP/s"))]
        self.assertEqual(bounds(*args),{"memory_s":Fraction(1,50),
            "compute_s":Fraction(1,125),"ideal_overlap_s":Fraction(1,50),
            "serial_resource_s":Fraction(7,250)})
        with self.assertRaises(ValueError):
            bounds(*args[:2], args[3], args[2])
        with self.assertRaises(ValueError):
            bounds(*args[:2],Quantity.of(0,"byte/s"),args[3])

    def test_different_limits_change_the_active_resource(self):
        slow = bounds(Quantity.of(100,"byte"),Quantity.of(1000,"FLOP"),
                      Quantity.of(100,"byte/s"),Quantity.of(1000,"FLOP/s"))
        fast_memory = bounds(Quantity.of(50,"byte"),Quantity.of(1000,"FLOP"),
                             Quantity.of(100,"byte/s"),Quantity.of(1000,"FLOP/s"))
        self.assertEqual(slow["ideal_overlap_s"], fast_memory["ideal_overlap_s"])
        self.assertEqual(max(Fraction(20,4),8),8)  # traffic quartered
        self.assertEqual(Fraction(40,30),1/(1-Fraction(1,2)+Fraction(1,2)/2))

    def test_integer_contract_and_decimal_binary_units(self):
        self.assertEqual(((80+31)//32)*32,96)
        self.assertEqual(8*((10+31)//32)*32,256)
        self.assertEqual(80+128*2,336)
        self.assertNotEqual(10**9,2**30)
        self.assertEqual(Fraction(2**30,10**9),Fraction(2097152,1953125))
        for args in (([True],64,4),([-1],64,4),([1],64,0)):
            with self.assertRaises(ValueError):
                cache_bytes(*args)
        with self.assertRaises(ValueError):
            projection(1.5,16,8,80)


if __name__ == "__main__":
    unittest.main()
