"""Checks that the calculator reproduces Chapter 6's tables.

    python3 -m unittest discover -s code/labs/ch06-napkin
"""
import unittest

import napkin


class ChapterTables(unittest.TestCase):
    def setUp(self):
        self.h100, self.b200 = napkin.MACHINES["h100"], napkin.MACHINES["b200"]
        self.bf16 = napkin.Model(napkin.MODELS["qwen3-8b"], 2, 2)
        self.fp8 = napkin.Model(napkin.MODELS["qwen3-8b"], 1, 1)

    def step_ms(self, model, machine, batch):
        return round(napkin.step(model, machine, batch, 4096, 0.8, 0.6, 0.0)[0] * 1e3, 1)

    def test_the_bf16_h100_table(self):
        self.assertEqual([self.step_ms(self.bf16, self.h100, b) for b in (1, 8, 32, 64, 98)],
                         [5.9, 7.5, 12.9, 20.1, 27.7])
        self.assertEqual(napkin.b_max(self.bf16, self.h100, 4096, 4e9), 98)

    def test_fp8_and_the_b200(self):
        self.assertEqual(napkin.b_max(self.fp8, self.h100, 4096, 4e9), 224)
        self.assertEqual(napkin.b_max(self.fp8, self.b200, 4096, 4e9), 555)
        t, _ = napkin.step(self.fp8, self.b200, 555, 4096, 0.8, 0.6, 0.0)
        self.assertAlmostEqual(napkin.dollars_per_million(self.b200, 555, t), 0.09, places=2)

    def test_the_knee_does_not_move_with_precision(self):
        a = napkin.knee_context(self.bf16, self.h100, 0.8, 0.6)
        b = napkin.knee_context(self.fp8, self.h100, 0.8, 0.6)
        self.assertLess(abs(a - b), 1)  # FP8's peak is 1,979, not exactly 2 x 989
        self.assertTrue(460 < a < 480)

    def test_a_mixture_of_experts_reads_more_as_the_batch_grows(self):
        moe = napkin.Model(napkin.MODELS["qwen3-30b-a3b"], 2, 2)
        self.assertLess(moe.weight_bytes(1), moe.weight_bytes(16))
        # 3.04 B multiplied per token; Qwen's "3.3 B activated" also counts the input table
        self.assertAlmostEqual(moe.flops(0) / 2 / 1e9, 3.04, delta=0.01)


if __name__ == "__main__":
    unittest.main()
