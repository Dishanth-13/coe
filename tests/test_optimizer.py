"""
Task 5 regression tests: optimizer feasibility, honest fallback, rationale text,
and truthful benchmark / demo reporting.
"""

import contextlib
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vecta.config import (WorkloadProfile, IndexConfig, IndexFamily, MetricType, PerformancePrediction,
                          BenchmarkResult, FAISS_DEFAULT_HNSW, FAISS_DEFAULT_IVF)
from vecta.cost_model.surrogate_model import SurrogateCostModel
from vecta.tuner.optimizer import AutoTuningOptimizer
from vecta.benchmark.dataset_generator import BenchmarkDatasetGenerator
from vecta.benchmark.evaluator import BenchmarkEvaluator
from vecta.benchmark.baseline_comparer import BaselineComparer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class _TableCostModel:
    """Deterministic stand-in cost model: fixed (recall, p95 ms, memory MB) per configuration."""

    def __init__(self, table):
        self.table = table

    def predict_batch(self, workload, configs):
        preds = []
        for cfg in configs:
            recall, p95, mem = self.table[cfg.id_string]
            meets = (recall >= workload.target_recall, p95 <= workload.max_latency_ms,
                     mem <= workload.memory_budget_mb)
            preds.append(PerformancePrediction(
                predicted_recall=recall, predicted_latency_p50=p95 * 0.7, predicted_latency_p95=p95,
                predicted_latency_p99=p95 * 1.3, predicted_memory_mb=mem, predicted_build_time_s=0.1,
                is_feasible=all(meets), meets_recall_target=meets[0], meets_latency_target=meets[1],
                meets_memory_budget=meets[2]))
        return preds


class _FixedResultEvaluator:
    """Stand-in evaluator returning fixed (recall, p95 ms) per configuration id."""
    k = 10

    def __init__(self, rows):
        self.rows = rows

    def evaluate_config(self, config):
        recall, p95 = self.rows[config.id_string]
        return BenchmarkResult(config=config, recall_at_k=recall, latency_p50_ms=p95 * 0.8,
                               latency_p95_ms=p95, latency_p99_ms=p95 * 1.2, latency_mean_ms=p95 * 0.85,
                               throughput_qps=1000.0 / (p95 * 0.85), memory_footprint_mb=10.0,
                               index_build_time_s=0.1)


class TestOptimizerFeasibility(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.optimizer = AutoTuningOptimizer(SurrogateCostModel())
        cls.feasible_wl = WorkloadProfile(num_vectors=15000, dimension=128, target_recall=0.90,
                                          max_latency_ms=2.0, memory_budget_mb=128.0)
        cls.infeasible_wl = WorkloadProfile(num_vectors=1_000_000, dimension=768, target_recall=0.99,
                                            max_latency_ms=0.05, memory_budget_mb=10.0)
        cls.feasible_res = cls.optimizer.tune(cls.feasible_wl)
        cls.infeasible_res = cls.optimizer.tune(cls.infeasible_wl)

    def test_feasible_workload_reports_predicted_success(self):
        res = self.feasible_res
        self.assertTrue(res["sla_satisfied"])
        self.assertFalse(res["is_fallback"])
        self.assertEqual(res["sla_status"], "predicted_feasible")
        self.assertGreater(res["feasible_candidate_count"], 0)
        self.assertEqual(res["violated_constraints"], [])
        pred = res["recommended_prediction"]
        self.assertTrue(pred["is_feasible"])
        self.assertGreaterEqual(pred["predicted_recall"], self.feasible_wl.target_recall)
        self.assertLessEqual(pred["predicted_latency_p95"], self.feasible_wl.max_latency_ms)
        self.assertLessEqual(pred["predicted_memory_mb"], self.feasible_wl.memory_budget_mb)
        self.assertIn("predicted to meet every target", res["rationale"])

    def test_feasible_winner_is_best_utility_among_feasible(self):
        feasible_scores = [c["score"] for c in self.feasible_res["top_candidates"] if c["is_feasible"]]
        self.assertTrue(self.feasible_res["top_candidates"][0]["is_feasible"])
        self.assertEqual(self.feasible_res["winner_score"], max(feasible_scores))

    def test_infeasible_workload_uses_honest_fallback(self):
        res = self.infeasible_res
        self.assertFalse(res["sla_satisfied"])
        self.assertTrue(res["is_fallback"])
        self.assertEqual(res["sla_status"], "no_feasible_configuration")
        self.assertEqual(res["feasible_candidate_count"], 0)
        self.assertGreater(len(res["violated_constraints"]), 0)
        self.assertFalse(res["recommended_prediction"]["is_feasible"])
        self.assertIn("No configuration", res["rationale"])
        self.assertIn("SLA NOT met", res["rationale"])
        self.assertNotIn("It satisfies", res["rationale"])
        # Latency target must be printed with its real value (0.05 ms used to print as 0.1 ms)
        self.assertIn("0.05 ms", res["rationale"])

    def test_fallback_is_the_minimum_violation_candidate(self):
        """
        Hand-built scenario with fixed predictions (no learned model): nothing meets all targets
        (recall >= 0.90, p95 <= 1.0 ms, memory <= 100 MB). Expected fallback, computed by hand as
        the smallest total relative shortfall:
          A recall 0.70                 -> 0.222
          B recall 0.899, p95 1.05 ms   -> 0.0011 + 0.05 = 0.0511
          C recall 0.86                 -> 0.0444   <- closest overall
          D p95 3.0 ms                  -> 2.0
        B has the HIGHER utility score, so a utility-ranked pick (the pre-fix behaviour) chooses B.
        """
        table = {
            "A": (0.70, 0.5, 10.0), "B": (0.899, 1.05, 10.0),
            "C": (0.86, 0.5, 10.0), "D": (0.95, 3.0, 10.0),
        }
        configs = {name: IndexConfig(family=IndexFamily.HNSW, M=8 * (i + 1), name_override=name)
                   for i, name in enumerate(table)}
        by_id = {cfg.id_string: table[name] for name, cfg in configs.items()}
        wl = WorkloadProfile(num_vectors=10000, dimension=64, target_recall=0.90,
                             max_latency_ms=1.0, memory_budget_mb=100.0)
        opt = AutoTuningOptimizer(_TableCostModel(by_id))
        opt.generate_candidate_space = lambda workload, include_flat=False: list(configs.values())

        res = opt.tune(wl)

        self.assertEqual(res["recommended_config"]["id_string"], configs["C"].id_string)
        self.assertFalse(res["sla_satisfied"])
        self.assertTrue(res["is_fallback"])
        self.assertEqual(res["violated_constraints"], ["recall"])
        self.assertIn(configs["C"].id_string, res["rationale"])
        self.assertIn("recall 86.0% vs target >= 90.0%", res["rationale"])
        # Guard: the scenario really distinguishes min-violation from utility ranking
        preds = opt.cost_model.predict_batch(wl, [configs["B"], configs["C"]])
        self.assertGreater(opt.evaluate_utility(preds[0], wl), opt.evaluate_utility(preds[1], wl))

    def test_exactly_one_winner_point_and_feasibility_flags(self):
        for res in (self.feasible_res, self.infeasible_res):
            winners = [p for p in res["all_points"] if p["is_winner"]]
            self.assertEqual(len(winners), 1)
            self.assertEqual(winners[0]["id"], res["recommended_config"]["id_string"])
        self.assertTrue(any(p["is_feasible"] for p in self.feasible_res["all_points"]))
        self.assertFalse(any(p["is_feasible"] for p in self.infeasible_res["all_points"]))

    def test_feasibility_is_strict_without_tolerance_bands(self):
        model = self.optimizer.cost_model
        cfg = IndexConfig(family=IndexFamily.HNSW, M=16, efConstruction=64, efSearch=32)
        base = model.predict(WorkloadProfile(), cfg)
        # Target recall just above the prediction: the old 2-point tolerance called this feasible
        wl = WorkloadProfile(target_recall=min(1.0, base.predicted_recall + 0.01),
                             max_latency_ms=100.0, memory_budget_mb=10_000.0)
        pred = model.predict(wl, cfg)
        self.assertFalse(pred.meets_recall_target)
        self.assertFalse(pred.is_feasible)
        # Latency just below the prediction: the old 5% tolerance called this feasible
        wl = WorkloadProfile(target_recall=0.01, max_latency_ms=base.predicted_latency_p95 * 0.97,
                             memory_budget_mb=10_000.0)
        pred = model.predict(wl, cfg)
        self.assertFalse(pred.meets_latency_target)
        self.assertFalse(pred.is_feasible)


class TestHonestBenchmarkReporting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        db, q = BenchmarkDatasetGenerator.generate_gaussian_clusters(
            num_vectors=4000, dimension=32, num_clusters=8, num_queries=30, random_seed=3)
        cls.evaluator = BenchmarkEvaluator(db, q, metric=MetricType.L2, k=10)

    def test_latency_ratio_wording(self):
        self.assertEqual(BaselineComparer.describe_latency_ratio(2.0), "2.00x faster")
        self.assertEqual(BaselineComparer.describe_latency_ratio(0.25), "4.00x slower")

    VECTA_CFG = IndexConfig(family=IndexFamily.HNSW, M=64, efConstruction=200, efSearch=512)

    def _arena(self, vecta_p95, hnsw=(0.55, 0.10), ivf=(0.25, 0.05)):
        """Run the arena against fixed, deterministic benchmark results (no timing noise)."""
        rows = {
            IndexConfig(family=IndexFamily.FLAT).id_string: (1.0, 0.30),
            FAISS_DEFAULT_HNSW.id_string: hnsw,
            FAISS_DEFAULT_IVF.id_string: ivf,
            self.VECTA_CFG.id_string: (0.95, vecta_p95),
        }
        return BaselineComparer.run_head_to_head_arena(_FixedResultEvaluator(rows), self.VECTA_CFG)

    def test_slower_config_is_not_called_a_speedup(self):
        # Vecta 0.40 ms vs FAISS Default HNSW (the higher-recall default) 0.10 ms: 4x SLOWER.
        # The pre-fix verdict read "Vecta delivers 0.25x speedup".
        report = self._arena(vecta_p95=0.40)
        self.assertEqual(report["reference_default"], "FAISS Default HNSW")
        self.assertEqual(report["latency_comparison"], "4.00x slower")
        self.assertIn("p95 latency is 4.00x slower", report["summary_verdict"])
        self.assertNotIn("faster", report["summary_verdict"])
        self.assertNotIn("speedup", report["summary_verdict"])
        self.assertEqual(report["speedup_factor"], 0.25)
        self.assertEqual(report["latency_ratio_vs_default"], 4.0)

    def test_faster_config_is_reported_as_faster(self):
        report = self._arena(vecta_p95=0.05)
        self.assertEqual(report["latency_comparison"], "2.00x faster")
        self.assertIn("p95 latency is 2.00x faster", report["summary_verdict"])
        self.assertNotIn("slower", report["summary_verdict"])

    def test_reference_default_is_the_higher_recall_default(self):
        # IVF default has the higher recall here, so the comparison is against IVF (0.05 ms): 8x slower
        report = self._arena(vecta_p95=0.40, hnsw=(0.30, 0.10), ivf=(0.60, 0.05))
        self.assertEqual(report["reference_default"], "FAISS Default IVF")
        self.assertEqual(report["latency_comparison"], "8.00x slower")

    def test_ground_truth_row_is_measured_not_extrapolated(self):
        report = BaselineComparer.run_head_to_head_arena(
            self.evaluator, IndexConfig(family=IndexFamily.HNSW, M=16, efSearch=32))
        gt = report["results"]["ground_truth_flat"]
        self.assertEqual(gt["recall_at_k"], 1.0)
        for key in ["latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "index_build_time_s", "throughput_qps"]:
            self.assertIn(key, gt)
        self.assertLessEqual(gt["latency_p50_ms"], gt["latency_p95_ms"])
        self.assertLessEqual(gt["latency_p95_ms"], gt["latency_p99_ms"])
        self.assertEqual(gt["config"]["family"], "FLAT")

    def test_measured_sla_check_matches_measurements(self):
        cfg = IndexConfig(family=IndexFamily.IVF_FLAT, nlist=64, nprobe=1)
        wl = WorkloadProfile(num_vectors=4000, dimension=32, target_recall=0.999,
                             max_latency_ms=1000.0, memory_budget_mb=0.001)
        report = BaselineComparer.run_head_to_head_arena(self.evaluator, cfg, workload=wl)
        sla = report["sla_check"]
        vec = report["results"]["vecta_auto_tuned"]
        self.assertEqual(sla["recall"]["value"], vec["recall_at_k"])
        self.assertEqual(sla["recall"]["basis"], "measured")
        self.assertEqual(sla["recall"]["met"], vec["recall_at_k"] >= 0.999)
        self.assertTrue(sla["latency_p95_ms"]["met"])
        self.assertFalse(sla["memory_mb"]["met"])
        self.assertFalse(sla["all_met"])
        self.assertIn("MISSED", report["summary_verdict"])
        self.assertIn("Not all targets were met", report["summary_verdict"])


class TestDemoHonesty(unittest.TestCase):

    def test_demo_source_has_no_unconditional_success_messages(self):
        with open(os.path.join(REPO_ROOT, "demo.py"), encoding="utf-8") as f:
            src = f.read()
        self.assertNotIn("SLA Protected", src)
        self.assertNotIn("High spatial clustering", src)
        self.assertNotIn("Latency Speedup", src)
        self.assertNotIn("Memory Savings", src)

    def test_demo_output_reports_measured_outcomes(self):
        import demo
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            demo.main()
        out = " ".join(buf.getvalue().split())  # rich wraps long lines; normalize whitespace
        self.assertNotIn("SLA Protected", out)
        self.assertNotIn("speedup", out.lower())
        self.assertIn("Recall target", out)
        self.assertRegex(out, r"x (faster|slower)")
        self.assertRegex(out, r"Status: (within SLA|SLA BREACHED)|not enough queries")


if __name__ == "__main__":
    unittest.main()
