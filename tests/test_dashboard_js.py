"""
FIX 6 regression tests (dashboard side): run the real app.js under Node.js with a minimal fake DOM
(tests/js/dashboard_harness.js) and assert what the user would see. Skipped if Node.js is absent.
"""

import json
import os
import shutil
import subprocess
import unittest

HARNESS = os.path.join(os.path.dirname(__file__), "js", "dashboard_harness.js")
NODE = shutil.which("node")


@unittest.skipIf(NODE is None, "Node.js is not installed; dashboard behaviour tests need it")
class TestDashboardBehaviour(unittest.TestCase):

    def run_scenario(self, name):
        proc = subprocess.run([NODE, HARNESS, name], capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return json.loads(proc.stdout.strip().splitlines()[-1])

    def test_successful_run_renders_everything_and_clears_banner(self):
        s = self.run_scenario("success")
        self.assertEqual(s["calls"].count("/api/benchmark"), 1)  # automatic benchmark ran
        self.assertTrue(s["bannerHidden"])
        self.assertEqual(s["winnerTitle"], "HNSW(M=16, efC=64, efS=32)")
        self.assertEqual(s["verdict"], "VERDICT-TEXT")
        self.assertEqual(s["arenaRows"], 4)
        self.assertFalse(s["searchDisabled"])
        self.assertIn("Estimated index memory (not measured RAM)", s["memoryCellTitle"])
        self.assertIn("serialized index size: 7.30 MB", s["memoryCellTitle"])

    def test_size_cap_warning_stays_visible_after_run(self):
        s = self.run_scenario("warnings")
        self.assertFalse(s["bannerHidden"])
        self.assertIn("warning", s["bannerClass"].split())
        self.assertIn("exceeds the interactive limit", s["bannerText"])
        self.assertEqual(s["arenaRows"], 4)  # the run itself still completed

    def test_failed_profile_keeps_previous_dashboard_state(self):
        s = self.run_scenario("profile_fails_after_success")
        self.assertFalse(s["bannerHidden"])
        self.assertIn("error", s["bannerClass"].split())
        self.assertIn("less than or equal to 4096", s["bannerText"])
        self.assertIn("previous workload, recommendation and index are unchanged", s["bannerText"])
        # Previous (still valid on the server) results remain usable
        self.assertEqual(s["winnerTitle"], "HNSW(M=16, efC=64, efS=32)")
        self.assertEqual(s["verdict"], "VERDICT-TEXT")
        self.assertEqual(s["arenaRows"], 4)
        self.assertFalse(s["searchDisabled"])
        self.assertFalse(s["benchmarkDisabled"])
        self.assertFalse(s["runDisabled"])
        self.assertEqual(s["calls"].count("/api/tune"), 1)  # no tune after the rejected profile

    def test_failed_tune_after_new_profile_disables_index_actions(self):
        s = self.run_scenario("tune_fails_after_profile")
        self.assertIn("error", s["bannerClass"].split())
        self.assertIn("Index rebuild failed", s["bannerText"])
        self.assertEqual(s["winnerTitle"], "No recommendation")
        self.assertTrue(s["searchDisabled"])
        self.assertTrue(s["benchmarkDisabled"])
        self.assertFalse(s["runDisabled"])

    def test_measured_selection_shows_predicted_and_measured_values(self):
        s = self.run_scenario("measured_selection")
        self.assertEqual(s["winnerRecall"], "95.0%")              # predicted value stays the headline
        self.assertEqual(s["recallSub"], "Measured: 98.0%")         # measured value shown separately
        self.assertEqual(s["latencySub"], "Measured: 0.341 ms")
        self.assertIn("Measured index 13.0 MB within 128 MB budget", s["memorySub"])
        self.assertTrue(s["slaText"].startswith("Measured: meets all targets"))
        self.assertIn("8 real FAISS benchmarks", s["recommendationDesc"])

    def test_prediction_only_result_is_labelled_not_benchmarked(self):
        s = self.run_scenario("prediction_only")
        self.assertEqual(s["recallSub"], "Not benchmarked")       # no fake "± 0.0%" from null uncertainty
        self.assertEqual(s["latencySub"], "Not benchmarked")
        self.assertIn("not benchmarked", s["recommendationDesc"])
        self.assertTrue(s["slaText"].startswith("Predicted to meet all targets"))

    def test_unchanged_deployment_raises_no_alarm(self):
        s = self.run_scenario("deployment_unchanged")
        self.assertTrue(s["bannerHidden"])
        self.assertFalse(s["searchDisabled"])
        self.assertFalse(s["benchmarkDisabled"])

    def test_deployment_replaced_by_another_session_is_flagged(self):
        s = self.run_scenario("deployment_replaced")
        self.assertFalse(s["bannerHidden"])
        self.assertIn("warning", s["bannerClass"].split())
        self.assertIn("Another session changed the server's deployment", s["bannerText"])
        self.assertIn("it now serves Flat-Exact", s["bannerText"])
        self.assertIn("out of date", s["bannerText"])
        # Actions that would silently use the other session's index are disabled until re-run
        self.assertTrue(s["searchDisabled"])
        self.assertTrue(s["benchmarkDisabled"])
        self.assertFalse(s["runDisabled"])

    def test_deployment_removed_by_another_session_is_flagged(self):
        s = self.run_scenario("deployment_none")
        self.assertIn("no index is deployed now", s["bannerText"])
        self.assertTrue(s["searchDisabled"])

    def test_memory_label_uses_server_budget_not_slider(self):
        # Slider default is 128 MB; the server tuned against 50 MB and predicts 60 MB
        s = self.run_scenario("memory_label")
        self.assertEqual(s["memorySub"], "Exceeds 50 MB budget (predicted)")
        self.assertIn("warn", s["memorySubClass"].split())


if __name__ == "__main__":
    unittest.main()
