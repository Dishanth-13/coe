"""
Task 4 regression tests: COSINE metric must normalize database and query vectors
consistently and return true cosine-similarity neighbors.
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vecta.config import IndexConfig, IndexFamily, MetricType
from vecta.engine.faiss_wrapper import FaissIndexWrapper, l2_normalize_rows
from vecta.engine.dynamic_index import DynamicIndexManager
from vecta.benchmark.evaluator import BenchmarkEvaluator


def cosine_topk(db, queries, k):
    dbn = db / np.linalg.norm(db, axis=1, keepdims=True)
    qn = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    sims = qn @ dbn.T
    order = np.argsort(-sims, axis=1)[:, :k]
    return order, np.take_along_axis(sims, order, axis=1)


class TestCosineSimilarity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(7)
        cls.d = 32
        base = rng.randn(2000, cls.d).astype(np.float32)
        # Wildly varying norms: raw inner product and cosine rankings disagree on this data
        cls.db = base * rng.uniform(0.05, 20.0, size=(2000, 1)).astype(np.float32)
        cls.queries = (rng.randn(30, cls.d) * rng.uniform(0.1, 10.0, size=(30, 1))).astype(np.float32)
        cls.k = 10
        cls.gt_idx, cls.gt_sims = cosine_topk(cls.db.astype(np.float64), cls.queries.astype(np.float64), cls.k)

    def build(self, cfg, db=None):
        w = FaissIndexWrapper(cfg, self.d, MetricType.COSINE)
        w.build_and_add(self.db if db is None else db)
        return w

    def test_data_makes_cosine_differ_from_raw_inner_product(self):
        # Guard: the fixtures must be able to detect a missing normalization
        ip = FaissIndexWrapper(IndexConfig(family=IndexFamily.FLAT), self.d, MetricType.IP)
        ip.build_and_add(self.db)
        _, ip_idx, _ = ip.search(self.queries, k=self.k)
        self.assertLess(BenchmarkEvaluator.calculate_recall_at_k(ip_idx, self.gt_idx, self.k), 0.5)

    def test_flat_cosine_matches_exact_cosine_neighbors_and_scores(self):
        w = self.build(IndexConfig(family=IndexFamily.FLAT))
        sims, idx, _ = w.search(self.queries, k=self.k)
        self.assertEqual(BenchmarkEvaluator.calculate_recall_at_k(idx, self.gt_idx, self.k), 1.0)
        np.testing.assert_allclose(sims, self.gt_sims, atol=1e-4)
        self.assertTrue(np.all(sims <= 1.0 + 1e-5))

    def test_cosine_results_invariant_to_vector_scaling(self):
        w = self.build(IndexConfig(family=IndexFamily.FLAT))
        _, idx_a, _ = w.search(self.queries, k=self.k)
        _, idx_b, _ = w.search(self.queries * 123.0, k=self.k)
        np.testing.assert_array_equal(idx_a, idx_b)
        rescaled_db = self.db * np.random.RandomState(1).uniform(0.01, 50, size=(len(self.db), 1)).astype(np.float32)
        w2 = self.build(IndexConfig(family=IndexFamily.FLAT), db=rescaled_db)
        _, idx_c, _ = w2.search(self.queries, k=self.k)
        np.testing.assert_array_equal(idx_a, idx_c)

    def test_inputs_are_not_mutated(self):
        db_copy, q_copy = self.db.copy(), self.queries.copy()
        w = self.build(IndexConfig(family=IndexFamily.FLAT))
        w.search(self.queries, k=self.k)
        np.testing.assert_array_equal(self.db, db_copy)
        np.testing.assert_array_equal(self.queries, q_copy)

    def test_hnsw_cosine_high_recall_against_cosine_ground_truth(self):
        w = self.build(IndexConfig(family=IndexFamily.HNSW, M=32, efConstruction=200, efSearch=256))
        _, idx, _ = w.search(self.queries, k=self.k)
        self.assertGreaterEqual(BenchmarkEvaluator.calculate_recall_at_k(idx, self.gt_idx, self.k), 0.95)

    def test_ivf_flat_cosine_exhaustive_probe_is_exact(self):
        w = self.build(IndexConfig(family=IndexFamily.IVF_FLAT, nlist=16, nprobe=16))
        _, idx, _ = w.search(self.queries, k=self.k)
        self.assertEqual(BenchmarkEvaluator.calculate_recall_at_k(idx, self.gt_idx, self.k), 1.0)

    def test_ivf_pq_cosine_scores_are_cosine_scale(self):
        w = self.build(IndexConfig(family=IndexFamily.IVF_PQ, nlist=16, nprobe=16, m_pq=8))
        sims, idx, _ = w.search(self.queries, k=self.k)
        # PQ approximates unit vectors: scores must stay near the [-1, 1] cosine range, not raw IP magnitudes
        self.assertTrue(np.all(np.abs(sims) < 1.5))
        self.assertGreater(BenchmarkEvaluator.calculate_recall_at_k(idx, self.gt_idx, self.k), 0.3)

    def test_zero_query_vector_does_not_produce_nan(self):
        w = self.build(IndexConfig(family=IndexFamily.FLAT))
        sims, _, _ = w.search(np.zeros((1, self.d), dtype=np.float32), k=self.k)
        self.assertTrue(np.all(np.isfinite(sims)))
        np.testing.assert_allclose(l2_normalize_rows(np.zeros((2, 3))), np.zeros((2, 3)))

    def test_evaluator_ground_truth_uses_cosine(self):
        ev = BenchmarkEvaluator(self.db, self.queries, metric=MetricType.COSINE, k=self.k)
        self.assertEqual(BenchmarkEvaluator.calculate_recall_at_k(ev.gt_indices, self.gt_idx, self.k), 1.0)
        _, idx = ev.exact_search(self.queries[:3] * 9.0, k=self.k)
        np.testing.assert_array_equal(idx, self.gt_idx[:3])

    def test_dynamic_manager_cosine_query(self):
        mgr = DynamicIndexManager(self.d, MetricType.COSINE)
        mgr.set_active_index(self.build(IndexConfig(family=IndexFamily.FLAT)))
        res = mgr.query(self.queries[:1] * 0.001, k=self.k)
        self.assertEqual(res["indices"][0], self.gt_idx[0].tolist())


class TestCosineAPI(unittest.TestCase):
    """COSINE must be accepted by the API and searched with normalized queries."""

    def test_api_cosine_profile_tune_search(self):
        from fastapi.testclient import TestClient
        from vecta.api.app import app, state
        with TestClient(app) as client:
            wl = {"num_vectors": 5000, "dimension": 32, "metric": "cosine", "target_recall": 0.9,
                  "max_latency_ms": 2.0, "memory_budget_mb": 128.0, "target_k": 10}
            r = client.post("/api/profile", json=wl)
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(r.json()["workload_profile"]["metric"], "COSINE")
            self.assertEqual(client.post("/api/tune").status_code, 200)

            db = state.current_database
            query = (db[42] * 50.0).tolist()  # scaled: cosine must ignore the magnitude
            r = client.post("/api/search", json={"query_vector": query, "k": 10})
            self.assertEqual(r.status_code, 200, r.text)
            data = r.json()
            self.assertEqual(data["metric"], "COSINE")
            gt_idx, _ = cosine_topk(db.astype(np.float64), np.array([query], dtype=np.float64), 10)
            expected = len(set(gt_idx[0].tolist()).intersection(data["indices"][0]))
            self.assertEqual(data["exact_matches_in_top_k"], expected)
            # Query was scaled x50: un-normalized inner products would be ~50x larger. The deployed index
            # may be quantized (IVF-PQ), so scores are approximate cosines and may overshoot 1.0 slightly.
            self.assertLess(max(abs(s) for s in data["distances"][0]), 1.5)


if __name__ == "__main__":
    unittest.main()
