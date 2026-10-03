from decimal import Decimal, localcontext
import itertools
import math
import unittest
from unittest.mock import patch

import model as operators
from decoder_trace import chunk_attention, demonstrate, generation, residual_trace, tiled_swiglu
from model import Config, Decoder, attention_row, swiglu


def decimal_attention(q, keys, values):
    """Independent row oracle: Decimal arithmetic, no model primitives."""
    with localcontext() as ctx:
        ctx.prec = 70
        d = lambda x: Decimal(str(x))
        scores = [sum((d(a)*d(b) for a,b in zip(q,k)), Decimal(0))
                  / Decimal(len(q)).sqrt() for k in keys]
        exps = [(s-max(scores)).exp() for s in scores]
        weights = [e/sum(exps) for e in exps]
        return [float(sum(p*d(v[j]) for p,v in zip(weights,values)))
                for j in range(len(values[0]))]


class DecoderTrace(unittest.TestCase):
    def close(self, a, b, tolerance=1e-12):
        self.assertEqual(len(a),len(b))
        for x,y in zip(a,b):
            self.assertAlmostEqual(x,y,delta=tolerance)

    def test_absolute_chunk_mask_and_wrong_local_triangle(self):
        rows = chunk_attention([[0.,0.]]*2, [[1.,2.]]*5,
                               [[float(i)] for i in range(5)], 3)
        self.assertEqual([r["permitted"] for r in rows],[[0,1,2,3],[0,1,2,3,4]])
        self.assertEqual([r["output"] for r in rows],[[1.5],[2.]])
        # Wrong upper-left triangle admits only local+1 keys.
        wrong = [attention_row([0.,0.],[[1.,2.]]*(i+1),
                               [[float(j)] for j in range(i+1)])[1] for i in range(2)]
        self.assertEqual(wrong,[[0.],[.5]])
        self.assertNotEqual(wrong,[r["output"] for r in rows])

    def test_decimal_oracle_for_asymmetric_rows(self):
        for width in (2,4,6):
            q = [(-1)**i*(i+1)/4 for i in range(width)]
            keys = [[(i-j)/3 for i in range(width)] for j in range(4)]
            values = [[j+1., 2.-j, (-1.)**j] for j in range(4)]
            self.close(attention_row(q,keys,values)[1],decimal_attention(q,keys,values))

    def test_equal_scores_convex_bounds_and_value_translation(self):
        q,k,v = [1.,-1.], [[1.,1.],[2.,2.],[3.,3.]], [[-2.,4.],[1.,-1.],[7.,3.]]
        probabilities, out = attention_row(q,k,v)
        self.close(probabilities,[1/3]*3)
        self.close(out,[2.,2.])
        shifted = [[row[0]+10,row[1]-4] for row in v]
        self.close(attention_row(q,k,shifted)[1],[out[0]+10,out[1]-4])
        for j in range(2):
            self.assertLessEqual(min(row[j] for row in v),out[j])
            self.assertLessEqual(out[j],max(row[j] for row in v))

    def test_chunk_rejects_inconsistent_history(self):
        for prefix in (-1, .5, 2):
            with self.assertRaises(ValueError):
                chunk_attention([[0.,0.]],[[0.,0.]],[[1.]],prefix)

    def test_four_queries_map_to_two_kv_heads(self):
        model = Decoder(Config(width=8,heads=4,kv_heads=2,layers=1))
        projected = model._project(model._token(0),model.weights[0],0)
        seen = []
        original = operators.attention_row
        def record(q,k,v):
            seen.append((q,k,v))
            return original(q,k,v)
        with patch.object(operators,"attention_row",record):
            model.full([0])
        self.assertEqual([r[1][0] for r in seen],
                         [projected[1][0],projected[1][0],projected[1][1],projected[1][1]])
        self.assertEqual([r[0] for r in seen],projected[0])
        pairs = sum(len(r["permitted"]) for _ in range(4)
                    for r in chunk_attention([[0.,0.]]*2,[[1.,2.]]*5,
                                             [[float(i)] for i in range(5)],3))
        self.assertEqual(pairs,36)

    def test_matched_ffn_budget_by_entry_enumeration(self):
        two = [[[1]*6 for _ in range(12)],[[1]*12 for _ in range(6)]]
        gated = [[[1]*6 for _ in range(8)],[[1]*6 for _ in range(8)],
                 [[1]*8 for _ in range(6)]]
        count = lambda matrices: sum(len(row) for matrix in matrices for row in matrix)
        self.assertEqual(count(two),144)
        self.assertEqual(count(gated),144)

    def test_residual_trace_independent_coordinate_equations(self):
        t = residual_trace()
        self.close(t["attention_input"],[3/math.sqrt(13),4/math.sqrt(13)])
        self.assertEqual(t["r"],[4.,2.])
        a,b = 4/math.sqrt(10.5),2/math.sqrt(10.5)
        p0,p1 = a/(1+math.exp(-a))*(a+b), b/(1+math.exp(-b))*(a-b)
        self.close(t["product"],[p0,p1])
        self.close(t["next"],[4+p0-p1,2+.5*p0+p1])
        self.assertNotEqual(t["r"],[t["attention_input"][0]+1,t["attention_input"][1]-2])

    def test_swapping_gate_changes_function(self):
        identity, up = [[1.,0.],[0.,1.]], [[1.,1.],[1.,-1.]]
        self.assertNotEqual(swiglu([1.,-1.],identity,up,identity),
                            swiglu([1.,-1.],up,identity,identity))

    def test_hidden_tiling_matches_materialized_path_and_tail(self):
        # 20 width/tile cases; the five-wide hidden axis forces tails.
        for width in (2,3,4,7):
            x = [i/4-1 for i in range(width)]
            gate = [[(j-i)/8 for i in range(width)] for j in range(5)]
            up = [[(j+i+1)/9 for i in range(width)] for j in range(5)]
            down = [[(j-i)/7 for j in range(5)] for i in range(width)]
            for tile in (1,2,3,5,8):
                out, peak = tiled_swiglu(x,gate,up,down,tile)
                self.close(out,swiglu(x,gate,up,down))
                self.assertEqual(peak,min(tile,5))

    def test_hidden_tiling_rejections(self):
        for tile in (0,-1,1.5):
            with self.assertRaises(ValueError):
                tiled_swiglu([1.],[[1.]],[[1.]],[[1.]],tile)
        with self.assertRaises(ValueError):
            tiled_swiglu([1.],[[1.]],[[1.],[2.]],[[1.]],1)

    def test_zero_sublayers_independent_whole_decoder_oracle(self):
        model = Decoder()
        model.embedding[0] = [3.,4.,0.,0.]
        model.output = [[1.,0.,0.,0.],[0.,1.,0.,0.],[1.,-1.,0.,0.],
                        [0.,0.,1.,0.],[0.,0.,0.,1.]]
        for w in model.weights:
            w["o"] = [[0.]*4 for _ in range(4)]
            w["down"] = [[0.]*6 for _ in range(4)]
        scale = math.sqrt(25/4+1e-5)
        expected = [3/scale,4/scale,-1/scale,0.,0.]
        self.close(model.full([0,0,0])[-1],expected)
        cache = model.cache()
        for _ in range(3):
            self.close(cache.append(0),expected)

    def test_all_three_token_histories_in_three_head_configurations(self):
        for heads,kv,layers in ((1,1,1),(2,1,2),(2,2,3)):
            model = Decoder(Config(heads=heads,kv_heads=kv,layers=layers))
            for tokens in itertools.product(range(5),repeat=3):
                cache = model.cache()
                full = model.full(tokens)
                for i,token in enumerate(tokens):
                    self.close(cache.append(token),full[i])
                    self.assertEqual(cache.position,i+1)

    def test_position_reset_mutation_detected_after_first_token(self):
        model, tokens = Decoder(), [0,1,2]
        cache = model.cache()
        self.close(cache.append(0),model.full([0])[-1])
        cache.position = 0  # deliberately wrong; scratch state only
        result = cache.append(1)
        self.assertGreater(max(abs(a-b) for a,b in zip(result,model.full(tokens[:2])[-1])),1e-5)

    def test_layer_cache_values_are_not_interchangeable(self):
        cache = Decoder().cache()
        cache.append(0)
        self.assertNotEqual(cache.keys[0],cache.keys[1])
        self.assertNotEqual(cache.values[0],cache.values[1])

    def test_count_projection_rows_and_attention_pairs(self):
        model, tokens = Decoder(), [0,1,2,3,4]
        counts = {"pairs":0,"rows":0}
        original_attention, original_project = operators.attention_row, model._project
        def attention(q,k,v):
            counts["pairs"] += len(k)
            return original_attention(q,k,v)
        def project(x,w,p):
            counts["rows"] += 1
            return original_project(x,w,p)
        with patch.object(operators,"attention_row",attention), patch.object(model,"_project",project):
            for size in (3,4,5):
                model.full(tokens[:size])
            self.assertEqual(counts,{"pairs":124,"rows":24})
            counts.update(pairs=0,rows=0)
            cache = model.cache()
            for token in tokens:
                cache.append(token)
            self.assertEqual(counts,{"pairs":60,"rows":10})

    def test_generation_budget_end_and_evaluated_frontier(self):
        result = generation(budget=4)
        self.assertEqual([r["evaluated"] for r in result["events"]],[3,4,5,6])
        self.assertEqual(result["cache_layer_lengths"],[6,6])
        end = result["events"][0]["selected"]
        stopped = generation(budget=4,end=end)
        self.assertEqual(len(stopped["events"]),1)
        self.assertEqual(stopped["cache_position"],3)
        with self.assertRaises(ValueError):
            generation(budget=0)
        for invalid in (-1,5,1.5):
            with self.assertRaises(ValueError):
                generation(end=invalid)

    def test_parameter_count_and_trace_boundary(self):
        t = demonstrate()
        self.assertEqual(t["matrix_parameters"],280)
        self.assertEqual(t["layer_matrix_parameters"],120)
        self.assertEqual(t["layer_zero_position_two"]["position"],2)
        self.assertEqual(len(t["layer_zero_position_two"]["q"]),2)
        self.assertEqual(len(t["layer_zero_position_two"]["k"]),1)


if __name__ == "__main__":
    unittest.main()
