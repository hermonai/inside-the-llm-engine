"""Executable mirrors of the foundation examples; no model or GPU required.

These are illustrative calculations, not performance measurements. Source
anchors deliberately fail when a printed premise or result changes: update
the derivation and its test together, not one independently.
"""
from decimal import Decimal, localcontext
from pathlib import Path
import math
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "tex"


def softmax(scores):
    maximum = max(scores)
    weights = [math.exp(x - maximum) for x in scores]
    total = math.fsum(weights)
    return [x / total for x in weights]


def choose(probabilities, draw):
    if not 0 <= draw < 1:
        raise ValueError("draw must be in [0,1)")
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if draw < cumulative:
            return index
    raise ValueError("probabilities do not cover the draw")


class Foundations(unittest.TestCase):
    def test_worked_problem_minimum_not_ceiling(self):
        import importlib.util
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        spec = importlib.util.spec_from_file_location("textbook_contract", ROOT/"scripts/check-textbook.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with TemporaryDirectory() as scratch:
            tex = Path(scratch)
            (tex/"worked").mkdir()
            chapter = tex/"foundations/f13-numerics.tex"
            worked = tex/"worked/f13-numerics.tex"
            inclusion = r"\input{worked/f13-numerics.tex}"
            with patch.object(module, "TEX", tex), patch.object(module, "ROOT", tex):
                for count in (3,6):
                    worked.write_text((r"\textbf{Problem.} P \textbf{Solution.} S"+"\n")*count)
                    self.assertEqual(module.worked_problem_count(chapter,inclusion),count)
                for body in ((r"\textbf{Problem.} P \textbf{Solution.} S"+"\n")*2,
                             (r"\textbf{Problem.} P"+"\n")*6):
                    worked.write_text(body)
                    with self.assertRaises(SystemExit):
                        module.worked_problem_count(chapter,inclusion)

    def printed(self, name, *anchors):
        text = (TEX / "foundations" / name).read_text()
        for anchor in anchors:
            self.assertIn(anchor, text, f"{name}: printed contract changed")

    def test_all_chapters_have_contextual_bridges(self):
        main = (TEX / "inside-the-llm-engine.tex").read_text()
        chapters = re.findall(r"\\input\{(chapters/[^}]+)\}", main)
        self.assertEqual(len(chapters), 42)
        titles = set()
        for chapter in chapters:
            body = (TEX / chapter).read_text()
            matches = list(re.finditer(r"\\Foundations\{([^}]+)\}\{", body))
            self.assertEqual(len(matches), 1, chapter)
            bridge = matches[0]
            titles.add(bridge[1])
            self.assertLess(body.index(r"\OpeningSection"), bridge.start())
            self.assertLess(bridge.start(), body.index(r"\NapkinSection"))
            explanation = body[bridge.end():body.index(r"\NapkinSection")]
            self.assertGreater(len(explanation.split()), 60, chapter)
            self.assertRegex(explanation, r"\\ref\{|Appendix A\.[1-8]", chapter)
        self.assertEqual(len(titles), 42)

    def test_foundations_lead_the_reading_path(self):
        main = (TEX / "inside-the-llm-engine.tex").read_text()
        self.assertNotIn(r"\appendix", main)
        self.assertLess(main.index("foundations/f01"), main.index("chapters/ch01"))
        paths = sorted((TEX / "foundations").glob("f[0-9][0-9]-*.tex"))
        self.assertEqual(len(paths), 17)
        for path in paths:
            text = path.read_text()
            self.assertRegex(text, r"^% status: (FULL|VERIFIED)\n", path.name)
            self.assertTrue(r"\input{worked/" + path.stem in text, path.name)

    def test_prediction_and_sampling(self):
        self.printed("introduction.tex",
                     r"\mathbf{p}=[1/2,\;1/4,\;1/4]",
                     r"$[2/3,1/6,1/6]$", "$0.6$")
        p = softmax([math.log(2), 0, 0])
        self.assertEqual(p, [0.5, 0.25, 0.25])
        self.assertEqual(choose(p, .6), 1)
        self.assertEqual(choose(p, 0), 0)
        self.assertEqual(choose(p, .5), 1)
        self.assertEqual(choose(p, .75), 2)
        revised = softmax([math.log(4), 0, 0])
        for actual, expected in zip(revised, [2/3, 1/6, 1/6]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(choose(revised, .6), 0)
        with self.assertRaises(ValueError):
            choose(p, 1)

    def test_gpu_capacity_and_transfer(self):
        self.printed("f15-gpu-architecture.tex", "64 KiB", "24 KiB",
                     "16 KiB", "3.90625", "16 GiB/s")
        self.assertEqual(min(8, 64 // 24), 2)
        self.assertEqual(min(8, 64 // 16), 4)
        self.assertEqual(64 * 2**20 / (16 * 2**30) * 1000, 3.90625)

    def test_cpu_lines_and_tails(self):
        self.printed("f14-cpu-simd.tex", "64-byte", "four-byte",
                     "eight-lane", "nineteen", "three elements")
        self.assertEqual(64 // 4, 16)
        self.assertEqual(divmod(19, 8), (2, 3))
        for size in range(33):
            vectors, tail = divmod(size, 8)
            self.assertEqual(8 * vectors + tail, size)
            self.assertLess(tail, 8)

    def test_file_coordinates(self):
        self.printed("f12-model-formats.tex", "120-byte", "$[2,3]$",
                     "$[0,24)$", "$[128,152)$", "148$")
        origin = 8 + 120
        elements = 2 * 3
        self.assertEqual((origin, origin + elements * 4), (128, 152))
        self.assertEqual(origin + (1 * 3 + 2) * 4, 148)
        figure = (TEX / "figures/foundation-file-offsets.tex").read_text()
        self.assertIn("file offset 148", figure)
        self.assertIn("absolute interval $[128,152)$", figure)

    def test_stable_softmax(self):
        self.printed("f13-numerics.tex", "[1000,999]", "[0.731059,0.268941]")
        probabilities = softmax([1000, 999])
        self.assertAlmostEqual(probabilities[0], .731059, places=6)
        self.assertAlmostEqual(math.fsum(probabilities), 1)
        self.assertEqual(probabilities, softmax([1, 0]))

    def test_quantization_and_metadata(self):
        self.printed("f13-numerics.tex", "$s_q=0.5$", "$[-2,0,3]$",
                     "$[-1,0,1.5]$", "4.5 bits", "$6.5$")
        values = [-1.1, .2, 1.6]
        codes = [max(-7, min(7, round(v/.5))) for v in values]
        self.assertEqual(codes, [-2, 0, 3])
        reconstruction = [v * .5 for v in codes]
        self.assertEqual(reconstruction, [-1, 0, 1.5])
        self.assertTrue(all(abs(a-b) <= .25
                            for a, b in zip(values, reconstruction)))
        self.assertEqual(round(1.25 / .5), 2)
        self.assertEqual(10 - min(7, round(10 / .5)) * .5, 6.5)
        self.assertEqual((32 * 4 + 16) / 8, 18)
        self.assertEqual((32 * 4 + 16) / 32, 4.5)

    def test_nonassociative_reduction(self):
        self.printed("f13-numerics.tex", "three significant digits",
                     "$1000$, $1$, $-1000$")
        with localcontext() as context:
            context.prec = 3
            a, b, c = map(Decimal, ("1000", "1", "-1000"))
            self.assertEqual((a + b) + c, 0)
            self.assertEqual((a + c) + b, 1)

    def test_cache_payload_and_translation(self):
        self.printed("f16-notation.tex", "64", "960", "1536", "576",
                     "$(1,1)$", "$[7,2]$")
        per_token = 2 * 2 * 2 * 4 * 2
        logical = 3 * 5 * per_token
        reserved = 3 * math.ceil(5/4) * 4 * per_token
        self.assertEqual((per_token, logical, reserved), (64, 960, 1536))
        self.assertEqual(reserved - logical, 576)
        block, offset = divmod(5, 4)
        self.assertEqual(([7, 2][block], offset), (2, 1))
        for length in range(1, 33):
            reserved_tokens = math.ceil(length/4) * 4
            self.assertTrue(0 <= reserved_tokens-length < 4)

    def test_roofline_units(self):
        quantization = (TEX / "chapters/ch10-weight-quantization.tex").read_text()
        self.assertIn(r"P\,q_w/8", quantization)
        self.assertIn(r"$q_w=8b_w$", quantization)
        self.assertNotIn(r"P\,b_w/8", quantization)
        self.printed("f16-notation.tex", r"\max(20,8)", "200 output tokens/s",
                     "4 FLOP/byte", "10$ FLOP/byte")
        data, flops, bandwidth, compute = 2e9, 8e9, 100e9, 1e12
        bound = max(data/bandwidth, flops/compute)
        self.assertEqual(bound, .020)
        self.assertEqual(flops/data, 4)
        self.assertEqual(compute/bandwidth, 10)
        self.assertEqual(4/bound, 200)

    def test_delivery_timestamps(self):
        self.printed("f17-measurements.tex",
                     "$17-3=14$", "$(40-17)/(3-1)=11.5$", "$44-3=41$")
        intended, send, admission, terminal = 0, 3, 7, 44
        delivered = [17, 25, 40]
        self.assertEqual(delivered[0]-send, 14)
        self.assertEqual(delivered[0]-intended, 17)
        self.assertEqual(admission-send, 4)
        gaps = [b-a for a, b in zip(delivered, delivered[1:])]
        self.assertEqual(gaps, [8, 15])
        self.assertEqual(sum(gaps)/len(gaps), 11.5)
        self.assertEqual(terminal-send, 41)

    def test_chapter_two_attention_figure(self):
        # The old footer confused shifted and unshifted denominators.
        figure = (TEX / "figures/ch02-attention-worked.tex").read_text()
        self.assertIn(r"2e^{-0.5}+e^{-1}+1 \approx 2.581", figure)
        self.assertIn(r"e^{-1}/2.581 \approx 0.143", figure)
        denominator = 2 * math.exp(-.5) + math.exp(-1) + 1
        self.assertAlmostEqual(denominator, 2.581, places=3)
        p = softmax([.5, 0, 1, .5])
        for actual, expected in zip(p, [.235, .143, .387, .235]):
            self.assertAlmostEqual(actual, expected, places=3)
        values = [[2, 0, 1, 0], [0, 2, 0, 1],
                  [1, 1, 1, 1], [0, 0, 2, 2]]
        output = [sum(p[i] * values[i][j] for i in range(4))
                  for j in range(4)]
        for actual, expected in zip(output, [.857, .673, 1.092, 1]):
            self.assertAlmostEqual(actual, expected, places=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
