"""
FIX 6 regression tests (API side): memory admission control and failed operations leaving the
previous valid state intact.
"""

import importlib
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_api import APITestBase, SMALL_WORKLOAD, STATE_ATTRS

api_module = importlib.import_module("vecta.api.app")
state = api_module.state


class TestMemoryAdmissionControl(APITestBase):

    def test_estimate_is_monotonic_and_bounded(self):
        est = api_module.estimate_peak_memory_mb
        self.assertLess(est(30000, 384), est(30000, 768))
        self.assertLess(est(15000, 128), api_module.MAX_ESTIMATED_PEAK_MEMORY_MB)   # dashboard default
        self.assertLess(est(30000, 384), api_module.MAX_ESTIMATED_PEAK_MEMORY_MB)   # dashboard maximum
        self.assertGreater(est(30000, 4096), api_module.MAX_ESTIMATED_PEAK_MEMORY_MB)

    def test_request_over_total_memory_limit_is_rejected_without_changing_state(self):
        before = {name: getattr(state, name) for name in STATE_ATTRS}
        # Each field is individually valid (N <= 10M, D <= 4096) but the combination is too large
        r = self.client.post("/api/profile", json=dict(SMALL_WORKLOAD, num_vectors=30000, dimension=4096))
        self.assertEqual(r.status_code, 400, r.text)
        self.assertIn("Estimated peak memory", r.json()["detail"])
        # The 30k cap is applied before the estimate, so 50k requests are judged at 30k
        r = self.client.post("/api/profile", json=dict(SMALL_WORKLOAD, num_vectors=50000, dimension=2304))
        self.assertEqual(r.status_code, 400, r.text)
        self.assertIn("N=30000", r.json()["detail"])
        for name in STATE_ATTRS:
            self.assertIs(getattr(state, name), before[name], name)

    def test_accepted_profile_reports_its_estimate(self):
        r = self.client.post("/api/profile", json=SMALL_WORKLOAD)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertAlmostEqual(r.json()["estimated_peak_memory_mb"],
                               round(api_module.estimate_peak_memory_mb(5000, 64), 1))


class TestFailedOperationsKeepPreviousState(APITestBase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        r = cls.client.post("/api/profile", json=SMALL_WORKLOAD)
        assert r.status_code == 200, r.text
        r = cls.client.post("/api/tune")
        assert r.status_code == 200, r.text

    def _snapshot(self):
        return {name: getattr(state, name) for name in STATE_ATTRS}

    def test_failed_index_rebuild_keeps_config_and_index_consistent(self):
        before = self._snapshot()
        deployed_before = self.client.get("/api/system").json()["active_index"]["config"]["id_string"]

        class ExplodingWrapper(api_module.FaissIndexWrapper):
            def build_and_add(self, vectors):
                raise RuntimeError("simulated FAISS build failure")

        with mock.patch.object(api_module, "FaissIndexWrapper", ExplodingWrapper):
            r = self.client.post("/api/tune")
        self.assertEqual(r.status_code, 500, r.text)
        detail = r.json()["detail"]
        self.assertIn("Index rebuild failed", detail)
        self.assertIn("simulated FAISS build failure", detail)
        self.assertIn(f"previously deployed index {deployed_before} is still serving", detail)

        # Nothing was replaced: active_config, serving index and last result are the same objects
        for name in STATE_ATTRS:
            self.assertIs(getattr(state, name), before[name], name)
        # ...and they still describe each other
        self.assertEqual(self.client.get("/api/system").json()["active_index"]["config"]["id_string"],
                         deployed_before)
        self.assertEqual(self.client.post("/api/search", json={"k": 5}).status_code, 200)
        bench = self.client.post("/api/benchmark")
        self.assertEqual(bench.status_code, 200, bench.text)
        self.assertEqual(bench.json()["results"]["vecta_auto_tuned"]["config"]["id_string"], deployed_before)

    def test_failed_profile_keeps_previous_dataset_and_index(self):
        before = self._snapshot()
        with mock.patch.object(api_module.WorkloadProfiler, "profile",
                               side_effect=RuntimeError("simulated profiler failure")):
            r = self.client.post("/api/profile", json=dict(SMALL_WORKLOAD, dimension=32))
        self.assertEqual(r.status_code, 500, r.text)
        self.assertIn("Profiling failed", r.json()["detail"])
        self.assertIn("previous dataset and index were kept", r.json()["detail"])
        for name in STATE_ATTRS:
            self.assertIs(getattr(state, name), before[name], name)
        # The previous D=64 index still serves D=64 queries
        r = self.client.post("/api/search", json={"query_vector": [0.1] * 64, "use_random_query": False})
        self.assertEqual(r.status_code, 200, r.text)

    def test_tune_failure_after_new_profile_reports_no_active_index(self):
        r = self.client.post("/api/profile", json=dict(SMALL_WORKLOAD, dimension=32))
        self.assertEqual(r.status_code, 200)
        with mock.patch.object(state.optimizer, "tune", side_effect=RuntimeError("simulated optimizer failure")):
            r = self.client.post("/api/tune")
        self.assertEqual(r.status_code, 500, r.text)
        self.assertIn("no index is currently active", r.json()["detail"])
        self.assertIsNone(state.active_config)
        self.assertIsNone(state.dynamic_manager)
        self.assertEqual(self.client.post("/api/search", json={"k": 5}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
