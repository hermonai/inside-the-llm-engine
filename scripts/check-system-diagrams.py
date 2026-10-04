"""Guard native mechanism coverage and the logical fixtures drawn in TikZ.

This is not a UML validator or a performance test. Rendered pages still need
visual review, and implementation behavior needs its own executable tests.
"""
from collections import Counter
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "tex"
KINDS = {"sequence", "state", "activity", "component", "memory", "timeline"}


def source(name):
    return (TEX / "figures" / (name + ".tex")).read_text()


def coverage():
    paths = re.findall(r"\\input\{(chapters/[^}]+)\}",
                       (TEX / "inside-the-llm-engine.tex").read_text())
    counts = Counter()
    for path in paths:
        found = set()
        for name in re.findall(r"\\EngineFigure\{([^}]+)\}", (TEX / path).read_text()):
            body = source(name)
            marker = re.search(r"^% diagram-kind: (\S+)$", body, re.M)
            if marker:
                kind = marker[1]
                if kind not in KINDS:
                    raise ValueError(f"{name}: unknown diagram kind {kind}")
                if r"\begin{tikzpicture}" not in body:
                    raise ValueError(f"{name}: not native TikZ")
                found.add(kind)
        if not found:
            raise ValueError(f"{path}: no mechanism diagram; a plot is not enough")
        counts.update(found)
    return {"systems_chapters_with_mechanism": len(paths),
            "chapter_coverage_by_kind": dict(sorted(counts.items()))}


class DiagramContracts(unittest.TestCase):
    def test_ragged_batch_coordinates_match_grants(self):
        text = source("ch15-ragged-batch")
        arrays = re.findall(r"\\foreach \\x/\\lab in \{([^}]+)\}", text)
        values = [[int(pair.split("/")[1]) for pair in array.split(",")]
                  for array in arrays]
        self.assertEqual(values, [[0, 1, 2, 4], [3, 1, 0, 1]])
        self.assertEqual([b-a for a, b in zip(values[0], values[0][1:])],
                         [1, 1, 2])
        self.assertIn("$A_3$", text)
        self.assertIn("$B_1$", text)
        self.assertIn("$C_0$", text)
        self.assertIn("$C_1$", text)

    def test_chunk_trace_uses_counts_and_zero_based_positions(self):
        text = source("ch18-request-trace")
        prefix, budget = 6144, 1024
        first = prefix + budget - 64
        second = first + budget - 80
        self.assertEqual((first, second), (7104, 8048))
        self.assertIn("through 7103", text)
        self.assertIn("through 8047", text)
        self.assertIn("schedule 960", text)
        self.assertIn("schedule 944", text)

    def test_every_systems_chapter_has_a_mechanism(self):
        self.assertEqual(coverage()["systems_chapters_with_mechanism"], 42)

    def test_tree_mask_is_ancestry_not_array_order(self):
        text = source("sys-tree-mask")
        cells = re.search(r"\\foreach \\i/\\j in \{([0-9/,]+)\}", text)[1]
        actual = {tuple(map(int, cell.split("/"))) for cell in cells.split(",")}
        parent = {"A": "P", "B": "P", "C": "A", "D": "A", "E": "B", "F": "B"}
        columns = ["P", "A", "B", "C", "D", "E", "F"]
        expected = set()
        for row, node in enumerate(columns[1:]):
            while True:
                expected.add((row, columns.index(node)))
                if node == "P":
                    break
                node = parent[node]
        self.assertEqual(actual, expected)
        self.assertNotIn((3, 2), actual, "D must not attend to sibling B")

    def test_paged_ownership_and_tail_capacity(self):
        text = source("sys-paged-pool")
        requests = re.findall(
            r"Request ([AB])\\\\(\d+) logical tokens\\\\table: \$\[([0-9,]+)\]", text)
        refs = {int(block): int(count) for block, count in
                re.findall(r"Physical block (\d+): references? (\d+)", text)}
        self.assertEqual(len(requests), 2)
        self.assertIn("Four slots per block", text)
        owners = Counter()
        addresses = {}
        for name, length, table in requests:
            table = list(map(int, table.split(",")))
            length = int(length)
            self.assertEqual(len(table), (length + 3) // 4)
            owners.update(table)
            addresses[name] = [(table[p // 4], p % 4) for p in range(length)]
            tail = re.search(name + r": \$\[([^]]+)\]", text)[1]
            self.assertEqual(tail.count(r"\text{unused}"), len(table) * 4 - length)
        self.assertEqual(dict(owners), refs)
        self.assertEqual(addresses["A"][:4], addresses["B"][:4])
        self.assertEqual(addresses["A"][4], (2, 0))
        self.assertEqual(addresses["B"][4], (9, 0))

    def test_retention_keeps_absolute_positions(self):
        text = source("sys-long-context-window")
        positions = list(map(int, re.search(r"\\foreach \\i in \{([0-9,]+)\}", text)[1].split(",")))
        self.assertEqual(positions, list(range(2)) + list(range(8, 12)))
        self.assertIn("$[0,1,8,9,10,11]$", text)
        self.assertIn("Never renumber retained positions for RoPE", text)

    def test_cache_element_counts_exclude_allocation_and_precision(self):
        text = source("sys-kv-architectures")
        for formula in ("$2H_qd_h$", "$2H_{kv}d_h$", "$d_c+d_R$"):
            self.assertIn(formula, text)
        # Independent illustrative dimensions, not measured memory footprints.
        queries, kv_heads, head_width, latent, rotary = 4, 2, 64, 96, 16
        self.assertEqual(2 * queries * head_width, 512)
        self.assertEqual(2 * kv_heads * head_width, 256)
        self.assertEqual(latent + rotary, 112)

    def test_rejection_residual_preserves_target_mass(self):
        p, q = [.5, .3, .2], [.1, .6, .3]
        accepted = [min(pi, qi) for pi, qi in zip(p, q)]
        rejection = 1 - sum(accepted)
        residual = [max(0, pi - qi) / rejection for pi, qi in zip(p, q)]
        for actual, target in zip(
                [a + rejection * r for a, r in zip(accepted, residual)], p):
            self.assertAlmostEqual(actual, target)
        naive = [a + rejection * pi for a, pi in zip(accepted, p)]
        self.assertNotEqual(naive, p)
        chapter = (TEX / "chapters/ch22-speculative-decoding.tex").read_text()
        self.assertIn("normalized positive residual", chapter)
        self.assertIn("unadjusted target at this point would not preserve", chapter)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=1).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(DiagramContracts))
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(json.dumps(coverage()))
