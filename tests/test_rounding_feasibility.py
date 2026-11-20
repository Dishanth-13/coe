"""
FIX 3 regression tests: feasibility, violation score, constraint report and rationale must be
computed from the same full-precision predictions and must never contradict each other.
"""

import math
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vecta.config import WorkloadProfile, IndexConfig, IndexFamily
from vecta.cost_model.surrogate_model import SurrogateCostModel
from vecta.tuner.optimizer import AutoTuningOptimizer, format_against_target, _fmt_percent, _fmt_ms


class TestRecallBoundaryCase(unittest.TestCase):
    """Reproduces the review case: predicted recall 0.494197 vs target 0.4942."""

    @classmethod
    def setUpClass(cls):
        cls.model = SurrogateCostModel()
        cls.optimizer = AutoTuningOptimizer(cls.model)
        cls.cfg = IndexConfig(family=IndexFamily.HNSW, M=8, efConstruction=32, efSearch=8)
        permissive = WorkloadProfile(num_vectors=15000, dimension=128, target_recall=0.01,
                                     max_latency_ms=1e6, memory_budget_mb=1e9)
        cls.raw_recall = cls.model.predict(permissive, cls.cfg).predicted_recall
        # Target = raw prediction rounded UP to 4 decimals (0.4942 for 0.494197): just out of reach
        cls.target = math.ceil(cls.raw_recall * 1e4) / 1e4
        cls.workload = WorkloadProfile(num_vectors=15000, dimension=128, target_recall=cls.target,
                                       max_latency_ms=1e6, memory_budget_mb=1e9)
        cls.pred = cls.model.predict(cls.workload, cls.cfg)

    def test_case_is_the_rounding_boundary(self):
        self.assertLess(self.raw_recall, self.target)
        self.assertLess(self.target - self.raw_recall, 1e-4)

    def test_prediction_keeps_full_precision(self):
        # Before the fix the stored value was round(raw, 4) == target, hiding the shortfall
        self.assertEqual(self.pred.predicted_recall, self.raw_recall)
        self.assertLess(self.pred.predicted_recall, self.workload.target_recall)

    def test_feasibility_violation_and_report_agree(self):
        self.assertFalse(self.pred.meets_recall_target)
        self.assertFalse(self.pred.is_feasible)
        self.assertGreater(self.optimizer.constraint_violation(self.pred, self.workload), 0.0)
        recall_row = AutoTuningOptimizer.describe_constraints(self.pred, self.workload)[0]
        self.assertFalse(recall_row["met"])
        self.assertLess(recall_row["predicted"], recall_row["target"])

    def test_rationale_shows_distinct_numbers_for_a_missed_target(self):
        text = AutoTuningOptimizer._build_rationale(self.cfg, self.pred, self.workload,
                                                    feasible_count=0, total_count=1)
        m = re.search(r"recall ([\d.]+)% vs target >= ([\d.]+)%", text)
        self.assertIsNotNone(m, text)
        shown_value, shown_target = m.group(1), m.group(2)
        self.assertNotEqual(shown_value, shown_target, text)       # was "49.4% vs 49.4%"
        self.assertLess(float(shown_value), float(shown_target))


class TestNoContradictionsAcrossGrid(unittest.TestCase):
    """Targets placed exactly at rounded prediction values create many boundary cases at once."""

    def test_flags_violation_and_report_consistent_for_every_candidate(self):
        model = SurrogateCostModel()
        opt = AutoTuningOptimizer(model)
        base = WorkloadProfile(num_vectors=15000, dimension=128)
        candidates = opt.generate_candidate_space(base)
        preds = model.predict_batch(base, candidates)
        checked = 0
        for probe in preds[::56]:
            wl = WorkloadProfile(num_vectors=15000, dimension=128,
                                 target_recall=round(probe.predicted_recall, 4),
                                 max_latency_ms=round(probe.predicted_latency_p95, 3),
                                 memory_budget_mb=round(probe.predicted_memory_mb, 2))
            for cfg in candidates:
                p = model.predict(wl, cfg)
                violation = opt.constraint_violation(p, wl)
                self.assertEqual(violation == 0.0, p.is_feasible, cfg.id_string)
                report = AutoTuningOptimizer.describe_constraints(p, wl)
                for row in report:
                    holds = (row["predicted"] >= row["target"]) if row["comparison"] == ">=" \
                        else (row["predicted"] <= row["target"])
                    self.assertEqual(row["met"], holds, (cfg.id_string, row))
                self.assertEqual(all(r["met"] for r in report), p.is_feasible)
                checked += 1
        self.assertGreater(checked, 1000)


class TestFormatting(unittest.TestCase):

    def test_unequal_values_never_print_identically(self):
        v, t = format_against_target(0.494197, 0.4942, _fmt_percent, 1)
        self.assertNotEqual(v, t)
        v, t = format_against_target(1.49999, 1.5, _fmt_ms, 3)
        self.assertNotEqual(v, t)

    def test_default_precision_kept_when_already_distinct_or_equal(self):
        self.assertEqual(format_against_target(0.946, 0.95, _fmt_percent, 1), ("94.6%", "95.0%"))
        self.assertEqual(format_against_target(0.95, 0.95, _fmt_percent, 1), ("95.0%", "95.0%"))
        self.assertEqual(format_against_target(0.34, 0.05, _fmt_ms, 3), ("0.34 ms", "0.05 ms"))


if __name__ == "__main__":
    unittest.main()
