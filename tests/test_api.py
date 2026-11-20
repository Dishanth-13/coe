"""
API regression tests: frontend/backend contract, validation and error handling.
Requires httpx (FastAPI TestClient).
"""

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from fastapi.testclient import TestClient
from vecta.api.app import app, state, _publish

# Shared serving state that endpoints replace (never mutate in place) when they succeed
STATE_ATTRS = ["current_database", "current_queries", "current_evaluator", "current_workload",
               "dynamic_manager", "active_config", "last_tuning_result"]


class StateIsolationMixin:
    """
    Snapshot the global API state before each test and restore it afterwards, even when the test
    fails, so one failing test cannot leave a different dataset / index behind for the next one.
    """

    def setUp(self):
        super().setUp()
        saved = {name: getattr(state, name) for name in STATE_ATTRS}
        self.addCleanup(lambda: _publish(**saved))

APP_JS = os.path.join(os.path.dirname(__file__), "..", "vecta", "ui", "static", "js", "app.js")
INDEX_HTML = os.path.join(os.path.dirname(__file__), "..", "vecta", "ui", "static", "index.html")

SMALL_WORKLOAD = {
    "num_vectors": 5000,
    "dimension": 64,
    "metric": "L2",
    "target_recall": 0.9,
    "max_latency_ms": 2.0,
    "memory_budget_mb": 128.0,
    "expected_qps": 500.0,
    "target_k": 10,
}


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


class APITestBase(StateIsolationMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._ctx = TestClient(app)
        cls.client = cls._ctx.__enter__()  # runs the startup event

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)

    def profile_and_tune(self, workload=None):
        wl = dict(SMALL_WORKLOAD, **(workload or {}))
        r = self.client.post("/api/profile", json=wl)
        self.assertEqual(r.status_code, 200, r.text)
        r = self.client.post("/api/tune")
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()


class TestFrontendContract(APITestBase):
    """Task 1: every field the dashboard reads must exist in the real API responses."""

    def test_tune_response_has_fields_used_by_frontend(self):
        tune = self.profile_and_tune()
        for key in ["recommended_config", "recommended_prediction", "rationale", "all_points",
                    "total_candidates_evaluated", "pareto_optimal_count", "tuning_duration_ms"]:
            self.assertIn(key, tune)
        pred = tune["recommended_prediction"]
        for key in ["predicted_recall", "recall_uncertainty", "predicted_latency_p95",
                    "latency_uncertainty_p95", "predicted_memory_mb", "predicted_build_time_s"]:
            self.assertIn(key, pred)
        self.assertIn("id_string", tune["recommended_config"])
        self.assertIn("family", tune["recommended_config"])
        point = tune["all_points"][0]
        for key in ["id", "recall", "latency_p95", "memory_mb", "is_pareto", "is_winner"]:
            self.assertIn(key, point)

    def test_benchmark_response_has_fields_used_by_frontend(self):
        self.profile_and_tune()
        r = self.client.post("/api/benchmark")
        self.assertEqual(r.status_code, 200, r.text)
        bench = r.json()
        self.assertIn("summary_verdict", bench)
        for key in ["vecta_auto_tuned", "faiss_default_hnsw", "faiss_default_ivf", "ground_truth_flat"]:
            self.assertIn(key, bench["results"])
            row = bench["results"][key]
            for field in ["recall_at_k", "latency_p95_ms", "memory_footprint_mb", "config"]:
                self.assertIn(field, row)

    def test_app_js_does_not_use_legacy_field_names(self):
        js = _read(APP_JS)
        for legacy in [r"pred\.recall\b", r"recall_ci_half_width", r"pred\.latency_p95_ms",
                       r"latency_ci_half_width", r"pred\.memory_mb", r"pred\.build_time_s",
                       r"selection_rationale", r"all_candidates", r"comparison_summary"]:
            self.assertIsNone(re.search(legacy, js), f"app.js still references legacy field {legacy}")


class TestDashboardRendering(APITestBase):
    """Task 2: dashboard markup and the data that populates it."""

    def test_every_element_id_used_by_app_js_exists_in_html(self):
        js = _read(APP_JS)
        html = _read(INDEX_HTML)
        ids = set(re.findall(r"getElementById\('([^']+)'\)", js))
        self.assertGreater(len(ids), 20)
        for element_id in ids:
            self.assertIn(f'id="{element_id}"', html, f"index.html is missing #{element_id} used by app.js")

    def test_html_has_no_hardcoded_demo_values(self):
        html = _read(INDEX_HTML)
        for placeholder in ["24 Threads", "8.9 GB", "FAISS AVX2", "312 Configurations", "50 Pareto Optimal",
                            "Tuned in 11.2 ms", "HNSW(M=48, efC=32, efS=128)", "94.6%", "0.22 ms",
                            "13.1 MB", "10 / 10 (100%)", "+37.6%", "Within memory limit", "Rapid indexing"]:
            self.assertNotIn(placeholder, html, f"hardcoded placeholder still present: {placeholder}")

    def test_html_has_status_and_empty_state_elements(self):
        html = _read(INDEX_HTML)
        for element_id in ["statusBanner", "paretoEmpty", "arenaEmpty"]:
            self.assertIn(f'id="{element_id}"', html)
        self.assertIn('class="empty-row"', html)

    def test_system_endpoint_reports_real_engine_and_hardware(self):
        r = self.client.get("/api/system")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        import faiss
        self.assertEqual(data["engine"]["name"], "FAISS")
        self.assertEqual(data["engine"]["version"], faiss.__version__)
        self.assertGreater(data["hardware"]["logical_cpus"], 0)

    def test_full_dashboard_flow_profile_tune_benchmark_search(self):
        self.profile_and_tune()
        r = self.client.post("/api/benchmark")
        self.assertEqual(r.status_code, 200, r.text)
        r = self.client.post("/api/search", json={"k": 10, "use_random_query": True})
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(len(data["indices"][0]), 10)


class TestAPIValidation(APITestBase):
    """Task 3: invalid input must produce meaningful HTTP 400s, never crashes or misleading results."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        r = cls.client.post("/api/profile", json=SMALL_WORKLOAD)
        assert r.status_code == 200, r.text
        r = cls.client.post("/api/tune")
        assert r.status_code == 200, r.text

    def assert_400(self, response, fragment=None):
        self.assertEqual(response.status_code, 400, response.text)
        detail = response.json()["detail"]
        self.assertIsInstance(detail, str)
        if fragment:
            self.assertIn(fragment, detail)

    def test_wrong_query_dimension_returns_400(self):
        r = self.client.post("/api/search", json={"query_vector": [0.1] * 5, "use_random_query": False})
        self.assert_400(r, "expects 64")

    def test_non_finite_query_vector_returns_400(self):
        body = '{"query_vector": [' + ", ".join(["NaN"] + ["0.1"] * 63) + '], "use_random_query": false}'
        r = self.client.post("/api/search", content=body, headers={"Content-Type": "application/json"})
        self.assert_400(r, "finite")

    def test_random_false_without_vector_returns_400_not_100_percent(self):
        r = self.client.post("/api/search", json={"k": 10, "use_random_query": False})
        self.assert_400(r, "requires a query_vector")

    def test_conflicting_query_options_return_400(self):
        r = self.client.post("/api/search", json={"query_vector": [0.1] * 64, "use_random_query": True})
        self.assert_400(r, "not both")

    def test_invalid_k_returns_400(self):
        self.assert_400(self.client.post("/api/search", json={"k": 0}))
        self.assert_400(self.client.post("/api/search", json={"k": "ten"}))

    def test_invalid_workload_fields_return_400(self):
        for bad in [{"metric": "MANHATTAN"}, {"target_recall": 1.5}, {"target_recall": 0},
                    {"dimension": 1}, {"max_latency_ms": -1}, {"memory_budget_mb": 0},
                    {"num_vectors": 10}, {"target_k": 0}]:
            r = self.client.post("/api/profile", json=dict(SMALL_WORKLOAD, **bad))
            self.assert_400(r)
        # malformed JSON body
        r = self.client.post("/api/profile", content="{not json", headers={"Content-Type": "application/json"})
        self.assert_400(r)

    def test_target_k_larger_than_dataset_returns_400(self):
        r = self.client.post("/api/profile", json=dict(SMALL_WORKLOAD, num_vectors=1000, target_k=1024))
        self.assert_400(r, "exceeds the dataset size")

    def test_custom_query_accuracy_matches_exact_brute_force(self):
        from vecta.api.app import state
        import numpy as np
        db = state.current_database
        query = db[123] + 0.01
        r = self.client.post("/api/search", json={"query_vector": query.tolist(), "k": 10})
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["query_source"], "custom")
        self.assertEqual(data["metric"], "L2")
        dists = np.sum((db - query.astype(np.float32)) ** 2, axis=1)
        exact = set(np.argsort(dists)[:10].tolist())
        expected_matches = len(exact.intersection(data["indices"][0]))
        self.assertEqual(data["exact_matches_in_top_k"], expected_matches)
        self.assertAlmostEqual(data["accuracy_pct"], expected_matches * 10.0)

    def test_search_k_larger_than_requested_ground_truth_k(self):
        """
        The evaluator stores ground truth for target_k=10 only. A k=25 search must be scored against
        the exact top-25 of the query actually used. (The old code compared against the stored
        10 columns, so it could never report more than 10 matches.)
        """
        self.assertEqual(state.current_evaluator.gt_indices.shape[1], 10)
        db, pool = state.current_database, state.current_queries
        saw_more_than_stored_k = False
        for _ in range(8):
            r = self.client.post("/api/search", json={"k": 25, "use_random_query": True})
            self.assertEqual(r.status_code, 200, r.text)
            data = r.json()
            self.assertEqual(data["k"], 25)
            # Identify the pool query that was used from the returned sample of its coordinates
            sample = np.array(data["query_vector_sample"], dtype=np.float32)
            used = np.where(np.all(np.isclose(pool[:, :8], sample, atol=1e-6), axis=1))[0]
            self.assertEqual(len(used), 1)
            query = pool[used[0]]
            exact_top25 = set(np.argsort(np.sum((db - query) ** 2, axis=1))[:25].tolist())
            expected = len(exact_top25.intersection(data["indices"][0]))
            self.assertEqual(data["exact_matches_in_top_k"], expected)
            self.assertAlmostEqual(data["accuracy_pct"], round(expected / 25 * 100, 1))
            saw_more_than_stored_k = saw_more_than_stored_k or expected > 10
        # Guard: the scenario must be able to expose the old 10-column comparison
        self.assertTrue(saw_more_than_stored_k)

    def test_dimension_change_without_retune_is_rejected(self):
        r = self.client.post("/api/profile", json=dict(SMALL_WORKLOAD, dimension=32))
        self.assertEqual(r.status_code, 200, r.text)
        # Index from the previous (D=64) dataset must not be used for search or benchmark
        self.assert_400(self.client.post("/api/search", json={"k": 10}), "Run /api/tune")
        self.assert_400(self.client.post("/api/benchmark"), "Run /api/tune")
        # After re-tuning, queries of the new dimension work
        self.assertEqual(self.client.post("/api/tune").status_code, 200)
        r = self.client.post("/api/search", json={"query_vector": [0.1] * 32, "use_random_query": False})
        self.assertEqual(r.status_code, 200, r.text)

    def test_oversized_workload_is_capped_with_explicit_warning(self):
        r = self.client.post("/api/profile", json=dict(SMALL_WORKLOAD, num_vectors=50000))
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["workload_profile"]["num_vectors"], 30000)
        self.assertTrue(any("interactive limit" in w for w in data["warnings"]))


class TestStateIsolation(APITestBase):
    """A test that fails after changing the shared API state must not leak that state."""

    def test_failed_test_does_not_leak_state(self):
        before = {name: getattr(state, name) for name in STATE_ATTRS}
        client = self.client

        class CorruptingTest(StateIsolationMixin, unittest.TestCase):
            def runTest(inner):
                r = client.post("/api/profile", json=dict(SMALL_WORKLOAD, dimension=32))
                inner.assertEqual(r.status_code, 200)
                inner.fail("simulated failure after changing the shared dataset")

        result = unittest.TestResult()
        CorruptingTest().run(result)
        self.assertEqual(len(result.failures), 1)
        for name in STATE_ATTRS:
            self.assertIs(getattr(state, name), before[name], name)
        # The pre-test dataset and index are usable again
        r = self.client.post("/api/search", json={"k": 5})
        self.assertEqual(r.status_code, 200, r.text)


if __name__ == "__main__":
    unittest.main()
