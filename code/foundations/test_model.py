import math
import struct
import unittest

from lesson import demonstrate, inspect_tensor, tensor_file
from model import Config, Decoder, attention_row, causal_attention, linear, rms_norm, rope, silu, softmax, swiglu


class Foundations(unittest.TestCase):
    def assert_vector_close(self, left, right, tolerance=1e-12):
        self.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            self.assertAlmostEqual(a, b, delta=tolerance)

    def test_attention_hand_calculation(self):
        output = demonstrate("attention")
        self.assert_vector_close(output["probabilities"], [.25, .75])
        self.assert_vector_close(output["output"], [.5, 3.])

    def test_mask_excludes_future_before_normalization(self):
        q = [[1., 0.], [1., 0.]]
        k = [[0., 0.], [1000., 0.]]
        v = [[2., 0.], [0., 1000.]]
        self.assertEqual(causal_attention(q,k,v)[0], [2.,0.])
        # Deliberately missing the mask must be caught by the same fixture.
        self.assertNotEqual(attention_row(q[0],k,v)[1], [2.,0.])

    def test_softmax_shift_and_rejections(self):
        self.assert_vector_close(softmax([1000.,999.]),softmax([1.,0.]))
        for invalid in ([], [math.inf], [math.nan], [-math.inf]):
            with self.assertRaises(ValueError):
                softmax(invalid)

    def test_attention_shapes_and_singleton(self):
        self.assertEqual(attention_row([1.,2.],[[4.,5.]],[[7.]])[1], [7.])
        for args in (([],[[]],[[1.]]), ([1.],[[1.,2.]],[[1.]]), ([1.],[[1.]],[])):
            with self.assertRaises(ValueError):
                attention_row(*args)

    def test_ffn_hand_calculation(self):
        output = demonstrate("ffn")
        self.assertEqual(output["relu_hidden"], [1.,0.,2.])
        self.assertEqual(output["relu_update"], [1.,2.])
        g = 3/(1+math.exp(-1))
        self.assert_vector_close(output["swiglu_update"],[-g,g])
        self.assertTrue(math.isfinite(silu(-1000)))

    def test_kernel_coordinates(self):
        self.assertEqual(linear([1.,2.],[[3.,4.],[5.,6.]]),[11.,17.])
        self.assert_vector_close(rms_norm([3.,4.],.5), [3/math.sqrt(13),4/math.sqrt(13)])
        self.assert_vector_close(rope([2.,3.],0),[2.,3.])
        self.assertAlmostEqual(sum(x*x for x in rope([2.,3.],5)),13.)

    def test_zero_gate_or_down_produces_no_update(self):
        zero, identity = [[0.,0.],[0.,0.]], [[1.,0.],[0.,1.]]
        self.assertEqual(swiglu([1.,-1.],zero,identity,identity),[0.,0.])
        self.assertEqual(swiglu([1.,-1.],identity,identity,zero),[0.,0.])

    def test_cached_vs_full_for_multiple_shapes_and_prefixes(self):
        for heads, kv_heads, layers in ((1,1,1),(2,1,2),(2,2,3)):
            model = Decoder(Config(heads=heads,kv_heads=kv_heads,layers=layers))
            cache = model.cache()
            tokens = [0,1,2,4,1,3]
            for i, token in enumerate(tokens,1):
                result = cache.append(token)
                self.assert_vector_close(result,model.full(tokens[:i])[-1])
                self.assertEqual(cache.position,i)
                self.assertTrue(all(len(k)==i for k in cache.keys))

    def test_decoder_future_cannot_change_past(self):
        model = Decoder()
        a,b = model.full([0,1,2]),model.full([0,1,4])
        for left,right in zip(a[:2],b[:2]):
            self.assert_vector_close(left,right)

    def test_printed_decoder_fixture(self):
        # Regression fixture, not an independent oracle for shared operators.
        expected = [.093927, -.204837, -.450096, -.577788, -.554559]
        result = Decoder().full([0,1,2])[-1]
        self.assert_vector_close(result, expected, tolerance=5e-7)
        self.assertEqual(max(range(len(result)), key=result.__getitem__), 0)

    def test_requests_have_independent_caches(self):
        model = Decoder()
        a,b = model.cache(),model.cache()
        a.append(0)
        b.append(4)
        self.assert_vector_close(a.append(1),model.full([0,1])[-1])
        self.assert_vector_close(b.append(2),model.full([4,2])[-1])

    def test_invalid_token_does_not_mutate_cache(self):
        model = Decoder()
        cache = model.cache()
        for token in (-1,5):
            with self.assertRaises(ValueError):
                cache.append(token)
        self.assertEqual(cache.position,0)
        self.assertTrue(all(not layer for layer in cache.keys))
        with self.assertRaises(ValueError):
            model.full([])
        with self.assertRaises(ValueError):
            Config(width=3)

    def test_generation_matches_recomputation(self):
        model = Decoder()
        prompt = [0,1,2]
        cache = model.cache()
        for token in prompt:
            logits = cache.append(token)
        for _ in range(5):
            expected = model.full(prompt)[-1]
            self.assert_vector_close(logits,expected)
            token = max(range(model.cfg.vocab),key=logits.__getitem__)
            prompt.append(token)
            logits = cache.append(token)

    def test_tensor_file_and_malformed_bounds(self):
        self.assertEqual(inspect_tensor(tensor_file()),
                         {"origin":128,"element_1_2":6.,"interval":[128,152]})
        for blob in (b"", tensor_file()[:-1], struct.pack("<Q", 2**63)+b"{}", tensor_file()+b"x"):
            with self.assertRaises(ValueError):
                inspect_tensor(blob)

    def test_remaining_chapter_examples(self):
        self.assertEqual(demonstrate("numerics")["codes"],[-2,0,3])
        self.assertEqual(demonstrate("cpu")["sum_of_1_to_19"],190)
        self.assertEqual(demonstrate("gpu")["transfer_ms"],3.90625)
        self.assertEqual(demonstrate("units")["reserved_bytes"],1536)
        self.assertEqual(demonstrate("measurement")["mean_gap_ms"],11.5)


if __name__ == "__main__":
    unittest.main()
