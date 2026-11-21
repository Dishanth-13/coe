"""
Tests for the predict -> shortlist -> benchmark -> select tuner (vecta/tuner/shortlist_tuner.py)
and its API integration.
"""

import importlib
import json
import math
import os
import sys
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vecta.config import IndexConfig, IndexFamily, MetricType, WorkloadProfile
from vecta.benchmark.dataset_generator import BenchmarkDatasetGenerator
from vecta.benchmark.evaluator import BenchmarkEvaluator
from vecta.cost_model import measured_model as mm
from vecta.cost_model.measured_model import MeasuredPrediction
from vecta.cost_model.memory_estimator import estimate_index_memory_mb
from vecta.tuner import shortlist_tuner as st
from vecta.tuner.shortlist_tuner import ShortlistTuner, shortlist_candidates
from vecta.tuner.optimizer import AutoTuningOptimizer

FLAT = IndexConfig(family=IndexFamily.FLAT)
IVF_1 = IndexConfig(family=IndexFamily.IVF_FLAT, nlist=16, nprobe=1)      # fast, low recall
IVF_16 = IndexConfig(family=IndexFamily.IVF_FLAT, nlist=16, nprobe=16)    # exhaustive, recall 1.0
HNSW_64 = IndexConfig(family=IndexFamily.HNSW, M=16, efConstruction=64, efSearch=64)
PQ_8 = IndexConfig(family=IndexFamily.IVF_PQ, nlist=8, nprobe=8, m_pq=8)
CANDIDATES = [FLAT, IVF_1, IVF_16, HNSW_64, PQ_8]


class ScriptedModel:
    """Stand-in cost model returning scripted (recall, p95) per configuration id."""

    def __init__(self, table, unsupported=(), fail=False):
        self.table, self.unsupported, self.fail = table, set(unsupported), fail

    def predict_batch(self, ds, configs):
        if self.fail:
            raise RuntimeError("scripted prediction failure")
        out = []
        for c in configs:
            if c.family.value in self.unsupported:
                out.append(None)
                continue
            recall, p95 = self.table[c.id_string]
            out.append(MeasuredPrediction(c.family.value, c.id_string, recall, p95 * 0.8, p95, 0.1,
                                          estimate_index_memory_mb(c.family, ds.num_vectors, ds.dimension, c), False))
        return out


def make_tuner(model=None, loader=None, size=3, **kw):
    synthetic = AutoTuningOptimizer()
    synthetic.generate_candidate_space = lambda workload, include_flat=False: list(CANDIDATES)
    if loader is None:
        loader = (lambda: (None, "no model in this test")) if model is None else None
    return ShortlistTuner(synthetic, measured_model=model, model_loader=loader, shortlist_size=size,
                          latency_passes=1, **kw)


class TunerTestBase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        db, q = BenchmarkDatasetGenerator.generate_gaussian_clusters(3000, 32, 8, num_queries=40, random_seed=5)
        cls.evaluator = BenchmarkEvaluator(db, q, metric=MetricType.L2, k=10)
        cls.fresh = {c.id_string: cls.evaluator.evaluate_config(c) for c in CANDIDATES}  # independent truth

    def workload(self, **kw):
        base = dict(num_vectors=3000, dimension=32, intrinsic_dimension=10.0, target_recall=0.9,
                    max_latency_ms=100.0, memory_budget_mb=100.0, target_k=10)
        return WorkloadProfile(**{**base, **kw})

    def scripted(self, *overrides):
        """Scripted model: (config, (predicted_recall, predicted_p95)) pairs; others default to (0.5, 1.0)."""
        table = {c.id_string: (0.5, 1.0) for c in CANDIDATES}
        table.update({cfg.id_string: value for cfg, value in overrides})
        return ScriptedModel(table)


class TestShortlisting(unittest.TestCase):

    def preds(self, rows):
        return [{"recall": r, "latency_p95_ms": l, "memory_mb": m, "undertrained": u} for r, l, m, u in rows]

    def test_order_exclusions_and_size(self):
        p = self.preds([
            (0.99, 0.50, 10, False),   # 0 predicted ok, slower
            (0.95, 0.10, 10, False),   # 1 predicted ok, fastest
            (0.80, 0.01, 10, False),   # 2 below target, higher recall
            (0.60, 0.01, 10, False),   # 3 below target, lower recall
            (0.99, 0.01, 500, False),  # 4 over memory budget
            (0.99, 0.01, 10, True),    # 5 under-trained
        ])
        chosen, excluded = shortlist_candidates(p, WorkloadProfile(target_recall=0.9, memory_budget_mb=100), size=3)
        self.assertEqual(chosen, [1, 0, 2])
        self.assertEqual(excluded, {4: "estimated index memory exceeds budget",
                                    5: "under-trained: too few k-means training points per centroid"})

    def test_family_diversity_cap_then_fill(self):
        p = self.preds([(0.99, 0.1 + i / 100, 10, False) for i in range(5)] + [(0.95, 0.9, 10, False)])
        fams = ["HNSW"] * 5 + ["IVF_FLAT"]
        chosen, _ = shortlist_candidates(p, WorkloadProfile(target_recall=0.9, memory_budget_mb=100), size=4,
                                         families=fams, max_per_family=2)
        self.assertEqual(chosen, [0, 1, 2, 5])  # 2 HNSW + IVF from the diversity pass, then filled with HNSW #3
        chosen, _ = shortlist_candidates(p, WorkloadProfile(target_recall=0.9, memory_budget_mb=100), size=4)
        self.assertEqual(chosen, [0, 1, 2, 3])  # no cap -> pure priority order

    def test_undertrained_within_budget_beats_over_budget(self):
        # Only an under-trained candidate fits the budget: it is shortlisted (flagged), the others are not
        p = self.preds([(0.99, 0.1, 300, False), (0.6, 0.1, 50, True), (0.99, 0.1, 200, False)])
        chosen, excluded = shortlist_candidates(p, WorkloadProfile(memory_budget_mb=100), size=3)
        self.assertEqual(chosen, [1])
        self.assertEqual(set(excluded), {0, 2})

    def test_everything_over_budget_shortlists_smallest(self):
        p = self.preds([(0.9, 0.1, 300, False), (0.9, 0.1, 200, False), (0.9, 0.1, 250, False)])
        chosen, excluded = shortlist_candidates(p, WorkloadProfile(memory_budget_mb=100), size=2)
        self.assertEqual(chosen, [1, 2])
        self.assertEqual(len(excluded), 3)

    def test_shortlist_size_validated(self):
        with self.assertRaises(ValueError):
            ShortlistTuner(shortlist_size=0)
        with self.assertRaises(ValueError):
            ShortlistTuner(shortlist_size=11)


class TestTieTolerance(unittest.TestCase):
    """select_measured_winner on hand-built measurements (no timing involved)."""

    def m(self, name, p95, mb, recall=1.0):
        return {"effective_config": name, "latency_p95_ms": p95, "serialized_index_mb": mb, "recall_at_k": recall}

    def run_select(self, rows, target=0.95, budget=100.0, max_latency=10.0, tol=0.05):
        measured = {r["effective_config"]: r for r in rows}
        checks = {k: {"recall": v["recall_at_k"] >= target, "memory": v["serialized_index_mb"] <= budget,
                      "latency": v["latency_p95_ms"] <= max_latency} for k, v in measured.items()}
        return st.select_measured_winner(measured, checks, tol)

    def test_near_tie_goes_to_smallest_index(self):
        winner, tie = self.run_select([self.m("HNSW", 0.376, 13.0), self.m("Flat", 0.385, 7.3),
                                       self.m("IVF", 0.386, 7.5)])
        self.assertEqual(winner, "Flat")
        self.assertTrue(tie["applied"])
        self.assertEqual(tie["fastest_config"], "HNSW")
        self.assertEqual(tie["tied_configs"], ["Flat", "HNSW", "IVF"])

    def test_clear_latency_win_is_kept(self):
        winner, tie = self.run_select([self.m("HNSW", 0.100, 13.0), self.m("Flat", 0.385, 7.3)])
        self.assertEqual(winner, "HNSW")
        self.assertFalse(tie["applied"])

    def test_zero_tolerance_is_plain_fastest(self):
        winner, _ = self.run_select([self.m("HNSW", 0.376, 13.0), self.m("Flat", 0.377, 7.3)], tol=0.0)
        self.assertEqual(winner, "HNSW")

    def test_infeasible_candidates_never_win_ties(self):
        winner, _ = self.run_select([self.m("HNSW", 0.376, 13.0), self.m("PQ", 0.380, 0.5, recall=0.6)])
        self.assertEqual(winner, "HNSW")

    def test_tie_break_does_not_cross_the_latency_limit(self):
        winner, _ = self.run_select([self.m("HNSW", 0.98, 13.0), self.m("Flat", 1.02, 7.3)], max_latency=1.0)
        self.assertEqual(winner, "HNSW")  # Flat is "tied" but would violate the 1.0 ms limit

    def test_stable_under_timing_noise(self):
        """Same three candidates, 200 noisy re-measurements (+-2%): the pick never changes."""
        rng = np.random.RandomState(0)
        winners = set()
        for _ in range(200):
            j = lambda v: v * (1 + rng.uniform(-0.02, 0.02))
            winners.add(self.run_select([self.m("HNSW", j(0.376), 13.0), self.m("Flat", j(0.380), 7.3),
                                         self.m("IVF", j(0.378), 7.5)])[0])
        self.assertEqual(winners, {"Flat"})
        # ...whereas picking the plain fastest flips between all of them
        plain = {self.run_select([self.m("HNSW", 0.376 * (1 + rng.uniform(-0.02, 0.02)), 13.0),
                                  self.m("Flat", 0.380 * (1 + rng.uniform(-0.02, 0.02)), 7.3),
                                  self.m("IVF", 0.378 * (1 + rng.uniform(-0.02, 0.02)), 7.5)], tol=0.0)[0]
                 for _ in range(200)}
        self.assertGreater(len(plain), 1)

    def test_negative_tolerance_rejected(self):
        with self.assertRaises(ValueError):
            ShortlistTuner(tie_tolerance=-0.1)


class TestMeasuredSelection(TunerTestBase):

    def test_final_choice_follows_measurements_not_predictions(self):
        # Prediction: fast IVF nprobe=1 meets the target -> it heads the shortlist. Reality: low recall.
        model = self.scripted((IVF_1, (0.99, 0.01)), (IVF_16, (0.95, 0.5)), (HNSW_64, (0.95, 0.4)))
        r = make_tuner(model).tune(self.workload(target_recall=0.9), self.evaluator)
        self.assertEqual(r["selection_basis"], "measured")
        self.assertEqual(r["shortlist"][0]["config"]["id_string"], IVF_1.id_string)   # predicted best
        winner = r["recommended_config"]["id_string"]
        self.assertNotEqual(winner, IVF_1.id_string)
        self.assertLess(self.fresh[IVF_1.id_string].recall_at_k, 0.9)                 # independent truth
        self.assertGreaterEqual(self.fresh[winner].recall_at_k, 0.9)
        self.assertAlmostEqual(r["recommended_measured"]["recall_at_k"], self.fresh[winner].recall_at_k, places=3)
        feasible = [e for e in r["shortlist"] if e["measured"] and e["meets"]["recall"] and e["meets"]["memory"]]
        fastest = min(e["measured"]["latency_p95_ms"] for e in feasible)
        tied = [e for e in feasible if e["measured"]["latency_p95_ms"] <= fastest * (1 + st.DEFAULT_TIE_TOLERANCE)]
        # Fastest feasible config, or - if several are within the timing-noise tolerance - the smallest index
        self.assertEqual(winner, min(tied, key=lambda e: (e["measured"]["serialized_index_mb"],
                                                          e["measured"]["effective_config"]))["config"]["id_string"])
        self.assertEqual(sorted(e["measured"]["effective_config"] for e in tied), r["tie_break"]["tied_configs"])
        self.assertTrue(r["sla_satisfied"])
        self.assertEqual(sum(e["selected"] for e in r["shortlist"]), 1)

    def test_recall_constraint_enforced_on_measurements(self):
        model = self.scripted((IVF_1, (0.999, 0.01)), (HNSW_64, (0.999, 0.02)), (IVF_16, (0.999, 0.5)))
        r = make_tuner(model).tune(self.workload(target_recall=0.999), self.evaluator)
        self.assertGreaterEqual(r["recommended_measured"]["recall_at_k"], 0.999)
        self.assertGreaterEqual(self.fresh[r["recommended_config"]["id_string"]].recall_at_k, 0.999)

    def test_memory_constraint_enforced(self):
        budget = 0.2  # MB: only IVF-PQ fits (3000 x 32 floats alone are 0.37 MB)
        model = self.scripted((PQ_8, (0.95, 0.1)), (FLAT, (1.0, 0.01)), (HNSW_64, (0.99, 0.01)))
        r = make_tuner(model).tune(self.workload(target_recall=0.1, memory_budget_mb=budget), self.evaluator)
        shortlisted = {e["config"]["id_string"] for e in r["shortlist"]}
        self.assertEqual(shortlisted, {PQ_8.id_string})  # memory-exceeding configs never benchmarked
        self.assertEqual(r["recommended_config"]["id_string"], PQ_8.id_string)
        # 3000 vectors are too few for a 256-centroid PQ codebook: flagged, but it is the only option in budget
        self.assertTrue(r["shortlist"][0]["predicted"]["undertrained"])
        self.assertLessEqual(r["recommended_measured"]["serialized_index_mb"], budget)
        self.assertGreater(r["excluded_counts"]["estimated index memory exceeds budget"], 0)

    def test_no_feasible_config_is_reported_as_best_effort(self):
        budget = 0.2
        model = self.scripted((PQ_8, (0.99, 0.1)))
        r = make_tuner(model).tune(self.workload(target_recall=0.999, memory_budget_mb=budget), self.evaluator)
        self.assertFalse(r["sla_satisfied"])
        self.assertEqual(r["sla_status"], "measured_best_effort")
        self.assertIn("recall", r["violated_constraints"])
        self.assertIn("SLA NOT met", r["rationale"])
        self.assertLessEqual(r["recommended_measured"]["serialized_index_mb"], budget)

    def test_flat_safety_net_when_shortlist_misses_recall(self):
        model = self.scripted((IVF_1, (0.99, 0.01)))
        r = make_tuner(model, size=1).tune(self.workload(target_recall=0.95), self.evaluator)
        roles = {e["config"]["id_string"]: e["role"] for e in r["shortlist"]}
        self.assertEqual(roles, {IVF_1.id_string: "shortlist", FLAT.id_string: "safety_net"})
        self.assertEqual(r["recommended_config"]["id_string"], FLAT.id_string)
        self.assertEqual(r["recommended_measured"]["recall_at_k"], 1.0)

    def test_variants_share_one_build_per_structure(self):
        builds = []
        real = st.FaissIndexWrapper.build_and_add

        def counting(self, vectors):
            builds.append(self.config.id_string)
            return real(self, vectors)

        model = self.scripted((IVF_1, (0.99, 0.01)), (IVF_16, (0.99, 0.02)))
        with mock.patch.object(st.FaissIndexWrapper, "build_and_add", counting):
            r = make_tuner(model, size=2).tune(self.workload(target_recall=0.5), self.evaluator)
        self.assertEqual(r["benchmarked_count"], 2)
        self.assertEqual(len(builds), 1)  # IVF nlist=16 built once, measured at nprobe 1 and 16

    def test_response_separates_predicted_and_measured_and_is_json(self):
        model = self.scripted((IVF_16, (0.97, 0.3)))
        r = make_tuner(model).tune(self.workload(), self.evaluator)
        json.dumps(r, allow_nan=False)
        self.assertEqual(r["recommended_prediction"]["source"], "measured_model")
        self.assertAlmostEqual(r["recommended_prediction"]["predicted_recall"],
                               model.table[r["recommended_config"]["id_string"]][0])
        for e in r["shortlist"]:
            self.assertIn("source", e["predicted"])
            self.assertTrue({"recall_at_k", "latency_p95_ms", "serialized_index_mb"} <= set(e["measured"]))
        for row in r["constraint_report"]:
            self.assertIn("measured", row["basis"])
            self.assertIn("predicted", row)
        pts = {p["id"]: p for p in r["all_points"]}
        self.assertTrue(all(p["measured_recall"] is not None for p in pts.values() if p["is_shortlisted"]))
        self.assertTrue(all(p["measured_recall"] is None for p in pts.values() if not p["is_shortlisted"]))


class TestFallbacks(TunerTestBase):

    def test_model_unavailable_uses_synthetic_predictions_but_still_measures(self):
        r = make_tuner(loader=lambda: (None, "no benchmark data")).tune(self.workload(), self.evaluator)
        self.assertEqual(r["prediction_model"], "synthetic_model")
        self.assertEqual(r["prediction_sources"], {"measured_model": 0, "synthetic_model": len(CANDIDATES)})
        self.assertEqual(r["fallback_reason"], "no benchmark data")
        self.assertEqual(r["selection_basis"], "measured")
        self.assertIsNotNone(r["recommended_measured"])

    def test_loader_exception_and_prediction_failure_fall_back(self):
        def boom():
            raise OSError("disk unavailable")
        r = make_tuner(loader=boom).tune(self.workload(), self.evaluator)
        self.assertIn("loading failed", r["fallback_reason"])
        self.assertEqual(r["prediction_model"], "synthetic_model")
        r = make_tuner(ScriptedModel({}, fail=True)).tune(self.workload(), self.evaluator)
        self.assertIn("prediction failed", r["fallback_reason"])
        self.assertEqual(r["prediction_sources"]["measured_model"], 0)

    def test_unsupported_family_uses_synthetic_for_that_family_only(self):
        model = self.scripted()
        model.unsupported = {"IVF_PQ"}
        r = make_tuner(model).tune(self.workload(), self.evaluator)
        self.assertEqual(r["prediction_sources"], {"measured_model": len(CANDIDATES) - 1, "synthetic_model": 1})
        self.assertIn("IVF_PQ", r["fallback_reason"])
        self.assertEqual({p["source"] for p in r["all_points"] if p["family"] == "IVF_PQ"}, {"synthetic_model"})

    def test_no_dataset_is_prediction_only_and_labelled(self):
        r = make_tuner(self.scripted()).tune(self.workload(), evaluator=None)
        self.assertEqual(r["tuning_mode"], "prediction_only")
        self.assertEqual(r["selection_basis"], "predicted")
        self.assertIsNone(r["recommended_measured"])
        self.assertTrue(r["rationale"].startswith("[Prediction only"))

    def test_all_benchmarks_failing_falls_back_to_prediction_only(self):
        with mock.patch.object(st.FaissIndexWrapper, "build_and_add", side_effect=RuntimeError("FAISS down")):
            r = make_tuner(self.scripted()).tune(self.workload(), self.evaluator)
        self.assertEqual(r["tuning_mode"], "prediction_only")
        self.assertIn("every shortlisted benchmark failed", r["fallback_reason"])
        self.assertTrue(all("FAISS down" in e for e in r["benchmark_errors"].values()))

    def test_partial_benchmark_failure_keeps_measured_selection(self):
        real = st.FaissIndexWrapper.build_and_add

        def flaky(self, vectors):
            if self.config.family == IndexFamily.HNSW:
                raise RuntimeError("HNSW build failed")
            return real(self, vectors)

        model = self.scripted((HNSW_64, (0.99, 0.01)), (IVF_16, (0.99, 0.5)))
        with mock.patch.object(st.FaissIndexWrapper, "build_and_add", flaky):
            r = make_tuner(model).tune(self.workload(), self.evaluator)
        hnsw = next(e for e in r["shortlist"] if e["config"]["family"] == "HNSW")
        self.assertIsNone(hnsw["measured"])
        self.assertIn("HNSW build failed", hnsw["error"])
        self.assertEqual(r["selection_basis"], "measured")
        self.assertNotEqual(r["recommended_config"]["family"], "HNSW")


class TestModelLoading(unittest.TestCase):

    def test_disable_missing_and_bad_data(self):
        with mock.patch.dict(os.environ, {mm.ENV_DISABLE: "1"}):
            model, reason = mm.load_measured_model()
        self.assertIsNone(model)
        self.assertIn("disabled", reason)
        model, reason = mm.load_measured_model(os.path.join(tempfile.gettempdir(), "does-not-exist.csv"))
        self.assertIsNone(model)
        self.assertIn("not found", reason)
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("status\nok\n")
        try:
            model, reason = mm.load_measured_model(f.name)
            self.assertIsNone(model)
            self.assertIn("could not train", reason)
        finally:
            os.remove(f.name)

    def test_env_path_takes_priority(self):
        with mock.patch.dict(os.environ, {mm.ENV_MEASUREMENTS: "custom.csv"}):
            self.assertEqual(mm.default_measurements_path(), "custom.csv")


# ---------------------------------------------------------------------------
# API integration
# ---------------------------------------------------------------------------

from tests.test_api import APITestBase, SMALL_WORKLOAD  # noqa: E402

api_module = importlib.import_module("vecta.api.app")
state = api_module.state


class TestAPIShortlistTuning(APITestBase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        r = cls.client.post("/api/profile", json=SMALL_WORKLOAD)
        assert r.status_code == 200, r.text
        r = cls.client.post("/api/tune")
        assert r.status_code == 200, r.text
        cls.tune = r.json()

    def test_tune_response_contract(self):
        t = self.tune
        self.assertEqual(t["tuning_mode"], "predict_shortlist_benchmark")
        self.assertEqual(t["selection_basis"], "measured")
        self.assertIn(t["prediction_model"], ("measured_model", "synthetic_model"))
        self.assertGreaterEqual(t["shortlist_size"], 5)
        self.assertLessEqual(t["shortlist_size"], 10)
        self.assertEqual(t["benchmarked_count"], len(t["shortlist"]))
        selected = [e for e in t["shortlist"] if e["selected"]]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["config"]["id_string"], t["recommended_config"]["id_string"])
        self.assertEqual(selected[0]["measured"], t["recommended_measured"])
        # Fields the dashboard reads are still present
        for key in ["predicted_recall", "predicted_latency_p95", "predicted_memory_mb", "predicted_build_time_s"]:
            self.assertIn(key, t["recommended_prediction"])
        for key in ["recall_at_k", "latency_p95_ms", "serialized_index_mb"]:
            self.assertIn(key, t["recommended_measured"])

    def test_measured_values_are_real_and_deployed_config_is_the_winner(self):
        winner = api_module._config_from_dict(self.tune["recommended_config"])
        fresh = state.current_evaluator.evaluate_config(winner)
        self.assertAlmostEqual(self.tune["recommended_measured"]["recall_at_k"], fresh.recall_at_k, places=3)
        self.assertEqual(self.tune["deployed_config"]["id_string"], self.tune["recommended_config"]["id_string"])
        system = self.client.get("/api/system").json()
        self.assertEqual(system["active_index"]["config"]["id_string"], self.tune["deployed_config"]["id_string"])
        if self.tune["sla_satisfied"]:
            self.assertGreaterEqual(self.tune["recommended_measured"]["recall_at_k"], SMALL_WORKLOAD["target_recall"])
            self.assertLessEqual(self.tune["recommended_measured"]["serialized_index_mb"],
                                 SMALL_WORKLOAD["memory_budget_mb"])

    def test_recommendation_benchmark_and_deployment_stay_consistent(self):
        """After every tune: recommended == selected shortlist entry == deployed == serving == arena config."""
        deployment_ids = []
        for _ in range(2):
            r = self.client.post("/api/tune")
            self.assertEqual(r.status_code, 200, r.text)
            t = r.json()
            rec = t["recommended_config"]["id_string"]
            selected = [e for e in t["shortlist"] if e["selected"]]
            self.assertEqual([e["config"]["id_string"] for e in selected], [rec])
            self.assertEqual(selected[0]["measured"], t["recommended_measured"])
            self.assertEqual(t["deployed_config"]["id_string"], rec)
            active = self.client.get("/api/system").json()["active_index"]
            self.assertEqual(active["config"]["id_string"], rec)
            self.assertEqual(active["deployment_id"], t["deployment_id"])
            arena = self.client.post("/api/benchmark").json()["results"]["vecta_auto_tuned"]
            self.assertEqual(arena["config"]["id_string"], rec)
            self.assertEqual(state.last_tuning_result["deployment_id"], t["deployment_id"])
            deployment_ids.append(t["deployment_id"])
        self.assertEqual(deployment_ids[1], deployment_ids[0] + 1)

    def test_other_sessions_changes_are_visible_through_deployment_id(self):
        mine = self.client.post("/api/tune").json()["deployment_id"]
        other = self.client.post("/api/tune").json()["deployment_id"]          # "another tab" re-tunes
        self.assertNotEqual(self.client.get("/api/system").json()["active_index"]["deployment_id"], mine)
        self.assertEqual(self.client.get("/api/system").json()["active_index"]["deployment_id"], other)
        self.assertEqual(self.client.post("/api/profile", json=SMALL_WORKLOAD).status_code, 200)
        self.assertIsNone(self.client.get("/api/system").json()["active_index"])  # re-profiled: none deployed

    def test_api_falls_back_when_model_unavailable(self):
        tuner = state.optimizer
        saved = (tuner.measured_model, tuner.model_loader)
        self.addCleanup(lambda: setattr(tuner, "measured_model", saved[0]) or setattr(tuner, "model_loader", saved[1]))
        tuner.measured_model, tuner.model_loader = None, lambda: (None, "no benchmark measurements found")
        r = self.client.post("/api/tune")
        self.assertEqual(r.status_code, 200, r.text)
        t = r.json()
        self.assertEqual(t["prediction_model"], "synthetic_model")
        self.assertEqual(t["fallback_reason"], "no benchmark measurements found")
        self.assertEqual(t["selection_basis"], "measured")
        json.dumps(t, allow_nan=False)

    def test_concurrent_tunes_and_polling_keep_state_consistent(self):
        statuses = {"tune": [], "system": []}
        done = threading.Event()

        def tune():
            statuses["tune"].append(self.client.post("/api/tune").status_code)

        def poll():
            while not done.is_set():
                statuses["system"].append(self.client.get("/api/system").status_code)

        poller = threading.Thread(target=poll)
        poller.start()
        tuners = [threading.Thread(target=tune) for _ in range(2)]
        for t in tuners:
            t.start()
        for t in tuners:
            t.join(timeout=300)
        done.set()
        poller.join(timeout=30)
        self.assertEqual(statuses["tune"], [200, 200])
        self.assertTrue(statuses["system"])
        self.assertEqual(set(statuses["system"]), {200})
        # The serving index, active config and last result describe the same configuration
        deployed = state.last_tuning_result["deployed_config"]["id_string"]
        self.assertEqual(state.dynamic_manager.active_wrapper.config.id_string, deployed)
        self.assertEqual(api_module._config_from_dict(state.last_tuning_result["recommended_config"]).id_string,
                         state.active_config.id_string)

    def test_failed_tuning_keeps_previous_state(self):
        before = {n: getattr(state, n) for n in ("active_config", "dynamic_manager", "last_tuning_result")}
        with mock.patch.object(st.ShortlistTuner, "benchmark", side_effect=RuntimeError("benchmark crashed")):
            r = self.client.post("/api/tune")
        self.assertEqual(r.status_code, 500, r.text)
        self.assertIn("benchmark crashed", r.json()["detail"])
        for name, value in before.items():
            self.assertIs(getattr(state, name), value, name)
        self.assertEqual(self.client.post("/api/search", json={"k": 5}).status_code, 200)


if __name__ == "__main__":
    unittest.main()
