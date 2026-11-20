"""
FIX 2 regression tests: estimated index memory must never be presented as measured RAM, and the
measured serialized index size must be reported separately and correctly.
"""

import contextlib
import io
import os
import re
import sys
import unittest
import numpy as np
import faiss

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vecta.config import IndexConfig, IndexFamily, MetricType, WorkloadProfile
from vecta.engine.faiss_wrapper import FaissIndexWrapper
from vecta.benchmark.evaluator import BenchmarkEvaluator
from vecta.benchmark.baseline_comparer import BaselineComparer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


class TestSerializedSizeIsMeasured(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(4)
        cls.db = rng.randn(3000, 32).astype(np.float32)
        cls.q = rng.randn(20, 32).astype(np.float32)

    def test_serialized_size_equals_faiss_serialization(self):
        for cfg in [IndexConfig(family=IndexFamily.FLAT),
                    IndexConfig(family=IndexFamily.HNSW, M=16),
                    IndexConfig(family=IndexFamily.IVF_PQ, nlist=32, nprobe=4, m_pq=8)]:
            w = FaissIndexWrapper(cfg, 32)
            w.build_and_add(self.db)
            independent_mb = faiss.serialize_index(w.index).nbytes / (1024 * 1024)
            self.assertAlmostEqual(w.serialized_size_mb, independent_mb, places=6, msg=cfg.id_string)
            # The estimate is never smaller than the measured serialized size
            self.assertGreaterEqual(w.estimated_memory_mb + 0.005, w.serialized_size_mb)

    def test_benchmark_result_reports_both_quantities(self):
        ev = BenchmarkEvaluator(self.db, self.q, metric=MetricType.L2, k=5)
        res = ev.evaluate_config(IndexConfig(family=IndexFamily.FLAT))
        expected = (3000 * 32 * 4) / (1024 * 1024)  # flat payload; serialization adds a small header
        self.assertGreater(res.index_serialized_mb, expected)
        self.assertLess(res.index_serialized_mb, expected + 0.01)
        d = res.to_dict()
        self.assertIn("index_serialized_mb", d)
        self.assertIn("memory_footprint_mb", d)


class TestMemoryWording(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(5)
        cls.ev = BenchmarkEvaluator(rng.randn(2000, 16).astype(np.float32),
                                    rng.randn(15, 16).astype(np.float32), metric=MetricType.L2, k=5)
        wl = WorkloadProfile(num_vectors=2000, dimension=16, target_recall=0.5, max_latency_ms=100.0,
                             memory_budget_mb=50.0)
        cls.report = BaselineComparer.run_head_to_head_arena(
            cls.ev, IndexConfig(family=IndexFamily.HNSW, M=16, efSearch=32), workload=wl)

    def test_sla_check_states_basis_of_each_value(self):
        sla = self.report["sla_check"]
        self.assertEqual(sla["recall"]["basis"], "measured")
        self.assertEqual(sla["latency_p95_ms"]["basis"], "measured")
        self.assertEqual(sla["memory_mb"]["basis"], "estimated")
        self.assertIn("serialized_index_mb", sla["memory_mb"])

    def test_verdict_never_calls_memory_measured(self):
        verdict = self.report["summary_verdict"]
        self.assertIn("estimated index memory", verdict)
        # No sentence fragment may describe memory as measured
        for fragment in re.split(r"[.;]", verdict):
            if "memory" in fragment.lower():
                self.assertNotIn("measured", fragment.lower(), fragment)

    def test_dashboard_labels(self):
        html = _read("vecta/ui/static/index.html")
        self.assertNotIn("RAM Footprint", html)
        self.assertNotIn("Memory Footprint", html)
        self.assertIn("Est. Index Memory", html)
        self.assertIn("Predicted Memory", html)

    def test_demo_output_labels_memory_as_estimated(self):
        import demo
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            demo.main()
        out = " ".join(buf.getvalue().split())
        self.assertNotIn("RAM (MB)", out)
        self.assertIn("Est. Index Mem", out)
        self.assertRegex(out, r"Memory budget \(estimated index memory\) \(<= [\d.]+\): estimated [\d.]+")
        self.assertNotRegex(out, r"Memory budget[^•]*: measured")

    def test_system_endpoint_names_memory_fields_honestly(self):
        from fastapi.testclient import TestClient
        from vecta.api.app import app
        with TestClient(app) as client:
            active = client.get("/api/system").json()["active_index"]
        self.assertNotIn("memory_mb", active)
        self.assertIn("estimated_memory_mb", active)
        self.assertGreater(active["serialized_index_mb"], 0.0)


if __name__ == "__main__":
    unittest.main()
