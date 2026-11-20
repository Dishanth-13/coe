"""
Task 6 regression tests: the configuration FAISS actually builds, the configuration that is
reported, and the memory estimate must agree; adjustments must be explicit; the IVF-PQ grid
must respect INDEX_SEARCH_SPACE["m_pq_divisors"].
"""

import json
import os
import sys
import unittest
import numpy as np
import faiss

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vecta.config import (
    IndexConfig, IndexFamily, MetricType, WorkloadProfile, INDEX_SEARCH_SPACE,
    FAISS_DEFAULT_HNSW, FAISS_DEFAULT_IVF, max_ivf_nlist, largest_valid_pq_m, max_pq_nbits
)
from vecta.engine.faiss_wrapper import FaissIndexWrapper
from vecta.benchmark.evaluator import BenchmarkEvaluator
from vecta.tuner.optimizer import AutoTuningOptimizer
from vecta.cost_model.surrogate_model import SurrogateCostModel


def ivf(index):
    return faiss.downcast_index(faiss.extract_index_ivf(index))


class TestWrapperConfigConsistency(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(0)
        cls.d = 64
        cls.db = rng.randn(1000, cls.d).astype(np.float32)
        cls.queries = rng.randn(10, cls.d).astype(np.float32)

    def test_helpers(self):
        self.assertEqual(max_ivf_nlist(1000), 125)
        self.assertEqual(max_ivf_nlist(3), 1)
        self.assertEqual(largest_valid_pq_m(64, 12), 8)   # largest divisor <= requested, never larger
        self.assertEqual(largest_valid_pq_m(64, 16), 16)
        self.assertEqual(largest_valid_pq_m(30, 32), 30)
        self.assertEqual(largest_valid_pq_m(7, 4), 1)
        self.assertEqual(max_pq_nbits(1000, 8), 8)
        self.assertEqual(max_pq_nbits(200, 8), 7)

    def test_nlist_clamp_is_reported_and_matches_faiss(self):
        w = FaissIndexWrapper(IndexConfig(family=IndexFamily.IVF_FLAT, nlist=1024, nprobe=256), self.d)
        w.build_and_add(self.db)
        self.assertEqual(w.requested_config.nlist, 1024)
        self.assertEqual(w.config.nlist, 125)
        self.assertEqual(ivf(w.index).nlist, w.config.nlist)
        self.assertEqual(w.config.nprobe, 125)
        self.assertEqual(ivf(w.index).nprobe, w.config.nprobe)
        self.assertIn("nlist=125", w.config.id_string)
        self.assertTrue(any(a.startswith("nlist: 1024 -> 125") for a in w.config_adjustments))
        self.assertTrue(any(a.startswith("nprobe: 256 -> 125") for a in w.config_adjustments))
        # search keeps using the effective nprobe
        w.search(self.queries, k=5)
        self.assertEqual(ivf(w.index).nprobe, 125)

    def test_pq_m_adjustment_is_reported_and_matches_faiss(self):
        w = FaissIndexWrapper(IndexConfig(family=IndexFamily.IVF_PQ, nlist=16, nprobe=4, m_pq=12), self.d)
        w.build_and_add(self.db)
        self.assertEqual(w.requested_config.m_pq, 12)
        self.assertEqual(w.config.m_pq, 8)
        self.assertEqual(ivf(w.index).pq.M, 8)
        self.assertIn("M_pq=8", w.config.id_string)
        self.assertTrue(any(a.startswith("m_pq: 12 -> 8") for a in w.config_adjustments))

    def test_valid_config_has_no_adjustments(self):
        w = FaissIndexWrapper(IndexConfig(family=IndexFamily.IVF_PQ, nlist=16, nprobe=4, m_pq=16), self.d)
        w.build_and_add(self.db)
        self.assertEqual(w.config_adjustments, [])
        self.assertEqual(w.config.to_dict(), w.requested_config.to_dict())
        self.assertEqual(ivf(w.index).pq.M, 16)
        self.assertEqual(ivf(w.index).nlist, 16)

    def test_tiny_dataset_pq_nbits_adjusted_instead_of_crashing(self):
        small = self.db[:200]
        w = FaissIndexWrapper(IndexConfig(family=IndexFamily.IVF_PQ, nlist=8, nprobe=2, m_pq=8), self.d)
        w.build_and_add(small)
        self.assertEqual(w.config.nbits_per_idx, 7)
        self.assertEqual(ivf(w.index).pq.nbits, 7)
        self.assertTrue(any(a.startswith("nbits_per_idx: 8 -> 7") for a in w.config_adjustments))

    def test_memory_estimate_uses_effective_configuration(self):
        clamped = FaissIndexWrapper(IndexConfig(family=IndexFamily.IVF_FLAT, nlist=1024, nprobe=8), self.d)
        clamped.build_and_add(self.db)
        exact = FaissIndexWrapper(IndexConfig(family=IndexFamily.IVF_FLAT, nlist=125, nprobe=8), self.d)
        exact.build_and_add(self.db)
        # Same index is built, so the estimate must be the same (requested nlist=1024 must not leak in)
        self.assertEqual(clamped.estimated_memory_mb, exact.estimated_memory_mb)

        pq_adj = FaissIndexWrapper(IndexConfig(family=IndexFamily.IVF_PQ, nlist=16, nprobe=4, m_pq=12), self.d)
        pq_adj.build_and_add(self.db)
        pq_ok = FaissIndexWrapper(IndexConfig(family=IndexFamily.IVF_PQ, nlist=16, nprobe=4, m_pq=8), self.d)
        pq_ok.build_and_add(self.db)
        self.assertAlmostEqual(pq_adj.estimated_memory_mb, pq_ok.estimated_memory_mb, places=2)

    def test_caller_config_and_global_defaults_are_never_mutated(self):
        cfg = IndexConfig(family=IndexFamily.IVF_FLAT, nlist=1024, nprobe=256)
        w = FaissIndexWrapper(cfg, self.d)
        w.build_and_add(self.db)
        w.set_runtime_parameters(nprobe=2)
        self.assertEqual((cfg.nlist, cfg.nprobe), (1024, 256))

        hnsw = FaissIndexWrapper(FAISS_DEFAULT_HNSW, self.d)
        hnsw.build_and_add(self.db)
        hnsw.set_runtime_parameters(efSearch=200)
        self.assertEqual(FAISS_DEFAULT_HNSW.efSearch, 16)
        self.assertEqual(hnsw.config.efSearch, 200)
        self.assertEqual(FAISS_DEFAULT_IVF.nlist, 100)

    def test_runtime_nprobe_is_clamped_and_reported(self):
        w = FaissIndexWrapper(IndexConfig(family=IndexFamily.IVF_FLAT, nlist=32, nprobe=4), self.d)
        w.build_and_add(self.db)
        w.set_runtime_parameters(nprobe=500)
        self.assertEqual(w.config.nprobe, 32)
        self.assertEqual(ivf(w.index).nprobe, 32)
        self.assertTrue(any("requested 500 at runtime -> 32" in a for a in w.config_adjustments))

    def test_benchmark_result_reports_effective_and_requested_config(self):
        ev = BenchmarkEvaluator(self.db, self.queries, metric=MetricType.L2, k=5)
        res = ev.evaluate_config(IndexConfig(family=IndexFamily.IVF_PQ, nlist=1024, nprobe=8, m_pq=12))
        self.assertEqual(res.config.nlist, 125)
        self.assertEqual(res.config.m_pq, 8)
        self.assertEqual(res.requested_config.nlist, 1024)
        self.assertEqual(res.requested_config.m_pq, 12)
        self.assertEqual(len(res.config_adjustments), 2)
        d = json.loads(json.dumps(res.to_dict()))
        self.assertEqual(d["config"]["id_string"], "IVF-PQ(nlist=125, nprobe=8, M_pq=8)")
        self.assertEqual(d["requested_config"]["family"], "IVF_PQ")


class TestCandidateGrid(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.optimizer = AutoTuningOptimizer(SurrogateCostModel())

    def pq_m_values(self, workload):
        return {c.m_pq for c in self.optimizer.generate_candidate_space(workload) if c.family == IndexFamily.IVF_PQ}

    def test_pq_grid_uses_configured_divisors(self):
        configured = set(INDEX_SEARCH_SPACE[IndexFamily.IVF_PQ]["m_pq_divisors"])
        self.assertEqual(self.pq_m_values(WorkloadProfile(dimension=128)), {m for m in configured if 128 % m == 0})
        self.assertNotIn(2, self.pq_m_values(WorkloadProfile(dimension=128)))  # 2 is not in the configured space
        self.assertEqual(self.pq_m_values(WorkloadProfile(dimension=48)), {4, 8, 16})
        # No configured value divides 30: fall back to one valid divisor
        self.assertEqual(self.pq_m_values(WorkloadProfile(dimension=30)), {30})

    def test_pq_grid_follows_config_changes(self):
        space = INDEX_SEARCH_SPACE[IndexFamily.IVF_PQ]
        original = list(space["m_pq_divisors"])
        try:
            space["m_pq_divisors"] = [8]
            self.assertEqual(self.pq_m_values(WorkloadProfile(dimension=128)), {8})
        finally:
            space["m_pq_divisors"] = original

    def test_grid_only_contains_buildable_ivf_configs(self):
        wl = WorkloadProfile(num_vectors=2000, dimension=64)
        cap = max_ivf_nlist(2000)
        for cfg in self.optimizer.generate_candidate_space(wl):
            if cfg.family in (IndexFamily.IVF_FLAT, IndexFamily.IVF_PQ):
                self.assertLessEqual(cfg.nlist, cap)
                self.assertLessEqual(cfg.nprobe, cfg.nlist)
                if cfg.family == IndexFamily.IVF_PQ:
                    self.assertEqual(64 % cfg.m_pq, 0)

    def test_tuned_winner_builds_without_adjustment(self):
        rng = np.random.RandomState(1)
        db = rng.randn(3000, 32).astype(np.float32)
        wl = WorkloadProfile(num_vectors=3000, dimension=32, target_recall=0.5, max_latency_ms=5.0,
                             memory_budget_mb=0.5)  # tiny budget pushes the optimizer toward IVF-PQ
        res = self.optimizer.tune(wl)
        rec = res["recommended_config"]
        cfg = IndexConfig(family=IndexFamily(rec["family"]), M=rec["M"], efConstruction=rec["efConstruction"],
                          efSearch=rec["efSearch"], nlist=rec["nlist"], nprobe=rec["nprobe"], m_pq=rec["m_pq"])
        w = FaissIndexWrapper(cfg, 32)
        w.build_and_add(db)
        self.assertEqual(w.config_adjustments, [], f"winner {rec['id_string']} needed adjustments")
        self.assertEqual(w.config.id_string, rec["id_string"])


class TestAPIDeployedConfig(unittest.TestCase):

    def test_tune_reports_deployed_config_matching_active_index(self):
        from fastapi.testclient import TestClient
        from vecta.api.app import app
        with TestClient(app) as client:
            r = client.post("/api/profile", json={"num_vectors": 5000, "dimension": 64, "target_recall": 0.9,
                                                  "max_latency_ms": 2.0, "memory_budget_mb": 128.0})
            self.assertEqual(r.status_code, 200, r.text)
            tune = client.post("/api/tune").json()
            self.assertIn("deployed_config", tune)
            self.assertIsInstance(tune["config_adjustments"], list)
            system = client.get("/api/system").json()
            self.assertEqual(system["active_index"]["config"]["id_string"], tune["deployed_config"]["id_string"])
            bench = client.post("/api/benchmark").json()
            self.assertEqual(bench["results"]["vecta_auto_tuned"]["config"]["id_string"],
                             tune["deployed_config"]["id_string"])


if __name__ == "__main__":
    unittest.main()
