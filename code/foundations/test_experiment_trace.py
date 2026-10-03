from dataclasses import replace
from fractions import Fraction
import csv
import io
from pathlib import Path
import re
from statistics import median
from statistics import quantiles
import unittest
from unittest.mock import patch
import experiment_trace as exp


class ExperimentTrace(unittest.TestCase):
    def test_named_boundaries_and_terminal_not_a_token(self):
        r = exp.Request("x",0,3,(17,25,40),44)
        self.assertEqual(r.metrics(),{"client_delay_ns":3,
            "ttft_from_send_ns":14,"ttft_from_plan_ns":17,
            "gaps_ns":(8,15),"mean_gap_ns":Fraction(23,2),
            "terminal_from_send_ns":41})
        self.assertEqual(replace(r,terminal_ns=100).metrics()["mean_gap_ns"],Fraction(23,2))

    def test_one_token_and_failure_without_tokens(self):
        self.assertIsNone(exp.Request("one",0,0,(1,),2).metrics()["mean_gap_ns"])
        self.assertIsNone(exp.Request("fail",0,0,(),2,"error").metrics()["ttft_from_plan_ns"])
        with self.assertRaises(ValueError):
            exp.Request("empty",0,0,(),2)
        at_drain = exp.Request("boundary",0,0,(2,),2)
        result = exp.cohort([at_drain],0,2,2,2)
        self.assertEqual(result["wire_tokens"],result["valid_completed_tokens"])

    def test_correct_completion_and_slo_gate(self):
        result = exp.cohort(exp.fixture(),0,40_000_000,10_000_000,10_000_000)
        self.assertEqual(result,{"requests":4,"failed_requests":1,
            "incorrect_requests":1,"wire_tokens":8,"valid_completed_tokens":5,
            "good_tokens":3,"wire_tokens_per_s":200,
            "valid_completed_tokens_per_s":125,"good_tokens_per_s":75})
        # Raising the gap target admits B, not failed C or incorrect D.
        self.assertEqual(exp.cohort(exp.fixture(),0,40_000_000,10_000_000,18_000_000)["good_tokens"],5)

    def test_cannot_drop_delay_from_planned_arrival_slo(self):
        b = exp.fixture()[1]
        self.assertEqual(b.metrics()["ttft_from_send_ns"],4_000_000)
        self.assertEqual(b.metrics()["ttft_from_plan_ns"],10_000_000)
        self.assertEqual(exp.cohort([b],0,40_000_000,5_000_000,20_000_000)["good_tokens"],0)

    def test_trace_validation_and_clock_domains(self):
        r = exp.fixture()[0]
        for change in ({"send_ns":6_000_000},{"terminal_ns":1},
                       {"tokens_ns":(9,5)},{"planned_ns":True}):
            with self.assertRaises(ValueError):
                replace(r,**change)
        for requests in ([r,r],[r,replace(exp.fixture()[1],clock="other-client")]):
            with self.assertRaises(ValueError):
                exp.cohort(requests,0,40_000_000,10_000_000,10_000_000)
        with self.assertRaises(ValueError):
            exp.cohort(exp.fixture(),0,30_000_000,10_000_000,10_000_000)

    def test_quantile_method_against_standard_library(self):
        values = [9,10,10,11,60]
        self.assertEqual(exp.percentile(values,Fraction(95,100)),60)
        self.assertEqual(exp.percentile(values,Fraction(95,100),"inclusive-linear"),Fraction(251,5))
        self.assertAlmostEqual(float(exp.percentile(values,Fraction(95,100),"inclusive-linear")),
                               quantiles(values,n=100,method="inclusive")[94])
        self.assertEqual(exp.percentile([7],1),7)
        for args in (([],1),([1],2),([1],Fraction(1,2),"unknown")):
            with self.assertRaises(ValueError):
                exp.percentile(*args)

    def test_pooled_and_request_populations_differ(self):
        self.assertEqual(exp.percentile([1]*100+[100],Fraction(95,100)),1)
        self.assertEqual(exp.percentile([1,100],Fraction(95,100)),100)

    def test_open_and_closed_loop_queue_fixtures(self):
        service = [4,4,20,4,4]
        open_rows = exp.fcfs([0,4,8,12,16],service)
        closed_rows = exp.fcfs([0,4,8,28,32],service)
        self.assertEqual([end-start for start,_,end in open_rows],[4,4,20,20,20])
        self.assertEqual([end-start for start,_,end in closed_rows],[4,4,20,4,4])
        self.assertEqual([end for _,_,end in open_rows],[end for _,_,end in closed_rows])

    def test_integer_oracle_and_deliberate_corruption(self):
        a,b = exp.matrices(2)
        self.assertEqual(a,[[-5,-4],[-2,-1]])
        self.assertEqual(b,[[-3,-2],[-1,0]])
        self.assertEqual(exp.oracle(a,b),[[19,10],[7,4]])
        for n in range(1,12):
            a,b = exp.matrices(n)
            expected = exp.oracle(a,b)
            exp.checked(exp.baseline(a,b),expected)
            candidate = exp.packed_dot(a,exp.pack(b))
            exp.checked(candidate,expected)
            candidate[-1][-1] += 1
            with self.assertRaises(ValueError):
                exp.checked(candidate,expected)

    def test_measure_preserves_raw_pairs_and_fails_closed(self):
        result = exp.measure(3,3,0)
        self.assertEqual([r["variant"] for r in result["raw"]],
                         ["baseline","packed","packed","baseline","baseline","packed"])
        self.assertEqual(len(result["raw"]),6)
        for row in result["raw"]:
            self.assertEqual(row["total_ns"],row["setup_ns"]+row["compute_ns"])
        with patch.object(exp,"packed_dot",return_value=[[0]]):
            with self.assertRaises(ValueError):
                exp.measure(3,1,0)
        with self.assertRaises(ValueError):
            exp.measure(1000,1,0)

    def test_recorded_raw_times_reproduce_summaries(self):
        record = (Path(__file__).resolve().parents[2] /
                  "research/measurements/2026-10-03-foundation-experiments.md").read_text()
        blocks = re.findall(r"```csv\n(.*?)\n```",record,re.S)
        self.assertEqual(len(blocks),2)
        for block,medians in zip(blocks,((1490500,805834),(1501083,809334))):
            rows = list(csv.DictReader(io.StringIO(block)))
            self.assertEqual(len(rows),18)
            for row in rows:
                self.assertEqual(int(row["total_ns"]),
                                 int(row["setup_ns"])+int(row["compute_ns"]))
            for variant,expected in zip(("baseline","packed"),medians):
                observed = [int(r["total_ns"]) for r in rows if r["variant"]==variant]
                self.assertEqual(len(observed),9)
                self.assertEqual(median(observed),expected)


if __name__ == "__main__":
    unittest.main()
