"""
FIX 1 regression tests: /api/system must never fail while /api/profile and /api/tune swap the
shared serving state concurrently.
"""

import importlib
import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient

api_module = importlib.import_module("vecta.api.app")
from vecta.config import IndexConfig, IndexFamily, MetricType
from vecta.engine.dynamic_index import DynamicIndexManager
from vecta.engine.faiss_wrapper import FaissIndexWrapper


def _small_manager():
    import numpy as np
    db = np.random.RandomState(0).randn(200, 8).astype("float32")
    wrapper = FaissIndexWrapper(IndexConfig(family=IndexFamily.FLAT), 8, MetricType.L2)
    wrapper.build_and_add(db)
    manager = DynamicIndexManager(8, MetricType.L2)
    manager.set_active_index(wrapper)
    return manager


class _SwapAfterFirstRead:
    """
    Stand-in for the shared state that simulates a concurrent /api/profile landing between two
    reads: the first read of `dynamic_manager` returns a live manager, every later read None.
    """

    def __init__(self, real_state, manager):
        object.__setattr__(self, "_real", real_state)
        object.__setattr__(self, "_manager", manager)
        object.__setattr__(self, "reads", 0)

    def __getattr__(self, name):
        if name == "dynamic_manager":
            object.__setattr__(self, "reads", self.reads + 1)
            return self._manager if self.reads == 1 else None
        return getattr(self._real, name)


class TestSystemEndpointRace(unittest.TestCase):

    def test_system_reads_manager_once_even_if_swapped_mid_request(self):
        manager = _small_manager()
        proxy = _SwapAfterFirstRead(api_module.state, manager)
        original = api_module.state
        api_module.state = proxy
        try:
            result = api_module.get_system_telemetry()  # raised AttributeError before the fix
        finally:
            api_module.state = original
        self.assertEqual(proxy.reads, 1)
        self.assertEqual(result["active_index"]["total_vectors"], 200)
        self.assertEqual(result["active_index"]["queries_served"], 0)

    def test_stress_publish_vs_system_reads(self):
        """Real publish path (as used by profile/tune) racing the real /api/system handler."""
        manager = _small_manager()
        saved = (api_module.state.dynamic_manager,)
        old_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)  # force very frequent thread switches to widen race windows
        errors, ok_calls, swaps = [], [0], [0]
        stop_at = time.time() + 3.0

        def writer():
            while time.time() < stop_at:
                api_module._publish(dynamic_manager=None)      # what /api/profile publishes
                api_module._publish(dynamic_manager=manager)   # what /api/tune publishes
                swaps[0] += 1

        def reader():
            while time.time() < stop_at:
                try:
                    api_module.get_system_telemetry()
                    ok_calls[0] += 1
                except Exception as exc:  # pragma: no cover - failure path
                    errors.append(repr(exc))

        try:
            threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        finally:
            sys.setswitchinterval(old_interval)
            api_module._publish(dynamic_manager=saved[0])
        self.assertGreater(swaps[0], 1000)
        self.assertGreater(ok_calls[0], 100)
        self.assertEqual(errors[:3], [], f"{len(errors)} /api/system failures during concurrent swaps")


class TestConcurrentHTTPRequests(unittest.TestCase):

    def test_concurrent_system_profile_tune_search_never_500(self):
        workloads = [
            {"num_vectors": 1000, "dimension": 16, "target_recall": 0.9, "max_latency_ms": 2.0,
             "memory_budget_mb": 64.0, "target_k": 5},
            {"num_vectors": 1500, "dimension": 24, "target_recall": 0.9, "max_latency_ms": 2.0,
             "memory_budget_mb": 64.0, "target_k": 5},
        ]
        statuses = {"system": [], "profile": [], "tune": [], "search": []}
        done = threading.Event()

        with TestClient(api_module.app) as client:
            def mutator():
                try:
                    for i in range(4):
                        statuses["profile"].append(client.post("/api/profile", json=workloads[i % 2]).status_code)
                        statuses["tune"].append(client.post("/api/tune").status_code)
                finally:
                    done.set()

            def poll(path, method, key, body=None):
                while not done.is_set():
                    if method == "get":
                        statuses[key].append(client.get(path).status_code)
                    else:
                        statuses[key].append(client.post(path, json=body).status_code)

            threads = [
                threading.Thread(target=mutator),
                threading.Thread(target=poll, args=("/api/system", "get", "system")),
                threading.Thread(target=poll, args=("/api/system", "get", "system")),
                threading.Thread(target=poll, args=("/api/search", "post", "search", {"k": 5})),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=300)

        self.assertEqual(statuses["profile"], [200] * 4)
        self.assertEqual(statuses["tune"], [200] * 4)
        self.assertGreater(len(statuses["system"]), 10)
        self.assertEqual(set(statuses["system"]), {200})
        # A search may legitimately hit the window between profile and tune (400: no active index),
        # but must never fail with a server error.
        self.assertTrue(set(statuses["search"]) <= {200, 400}, statuses["search"])


if __name__ == "__main__":
    unittest.main()
