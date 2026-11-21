"""
Tests for the measured-data cost model, index-specific memory estimator and LODO evaluation.
A tiny two-dataset benchmark is collected once (a few seconds) and reused.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import faiss

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vecta.config import IndexConfig, IndexFamily
from vecta.engine.faiss_wrapper import FaissIndexWrapper
from vecta.benchmark import collector as col
from vecta.cost_model.memory_estimator import estimate_index_memory_mb, hnsw_neighbor_slots_per_node
from vecta.cost_model.measured_model import (MeasuredCostModel, DatasetFeatures, is_undertrained,
                                             config_from_row, FAISS_MIN_POINTS_PER_CENTROID)
from vecta.cost_model import evaluation as ev
from vecta.cost_model.surrogate_model import SurrogateCostModel

TMP = None
ROWS = None
CSV_PATH = None


def setUpModule():
    global TMP, ROWS, CSV_PATH
    TMP = tempfile.mkdtemp(prefix="vecta-model-")
    spec = col.CollectionSpec(
        name="model-test",
        datasets=[
            col.DatasetSpec("tiny-clusters", "gaussian_clusters", 3000, 32, 40, seed=11, params={"num_clusters": 8}),
            col.DatasetSpec("tiny-isotropic", "isotropic_gaussian", 3000, 32, 40, seed=12),
            col.DatasetSpec("tiny-semantic", "semantic_embeddings", 3000, 32, 40, seed=13, params={"intrinsic_dim": 8}),
        ],
        hnsw_builds=[[8, 32], [16, 32]], hnsw_ef_search=[16, 64],
        ivf_flat_nlist=[16, 128], ivf_flat_nprobe=[1, 4, 16],
        ivf_pq_nlist=[16], ivf_pq_m=[8, 16], ivf_pq_nprobe=[1, 4],
        latency_repeats=1,
    )
    col.run_collection(spec, output_dir=TMP, run_id="run", progress=None)
    CSV_PATH = os.path.join(TMP, "run", col.MEASUREMENTS_FILE)
    ROWS = col.load_measurements(CSV_PATH)


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


class TestMemoryEstimator(unittest.TestCase):
    """Independent check against faiss.serialize_index (not the collector's numbers)."""

    @classmethod
    def setUpClass(cls):
        cls.x = np.random.RandomState(0).randn(5000, 48).astype(np.float32)

    def check(self, cfg, tolerance):
        w = FaissIndexWrapper(cfg, 48)
        w.build_and_add(self.x)
        actual = faiss.serialize_index(w.index).nbytes / (1024 * 1024)
        estimate = estimate_index_memory_mb(cfg.family, 5000, 48, w.config)
        self.assertLess(abs(estimate / actual - 1), tolerance, cfg.id_string)

    def test_flat_ivf_pq_are_exact(self):
        self.check(IndexConfig(family=IndexFamily.FLAT), 0.001)
        self.check(IndexConfig(family=IndexFamily.IVF_FLAT, nlist=64, nprobe=4), 0.001)
        self.check(IndexConfig(family=IndexFamily.IVF_PQ, nlist=32, nprobe=4, m_pq=8), 0.002)

    def test_hnsw_expected_graph_size(self):
        for M in (8, 16, 32):
            self.check(IndexConfig(family=IndexFamily.HNSW, M=M, efConstruction=40), 0.01)
        self.assertAlmostEqual(hnsw_neighbor_slots_per_node(16), 32 + 16 / 15)


class TestUndertrainedFlag(unittest.TestCase):

    def test_flag_matches_faiss_warning(self):
        x = np.random.RandomState(1).randn(2000, 16).astype(np.float32)
        ds = DatasetFeatures(2000, 16, 10.0)
        for nlist in (16, 51, 52, 64):
            cfg = IndexConfig(family=IndexFamily.IVF_FLAT, nlist=nlist, nprobe=4)
            w = FaissIndexWrapper(cfg, 16)
            with col.capture_native_stderr() as cap:
                w.build_and_add(x)
            warned = "WARNING clustering" in cap.text
            self.assertEqual(is_undertrained(ds, cfg), warned, nlist)
        self.assertTrue(is_undertrained(DatasetFeatures(5000, 32, 8.0),
                                        IndexConfig(family=IndexFamily.IVF_PQ, nlist=8, m_pq=8)))  # 5000 < 39*256

    def test_flags_agree_with_recorded_warnings_in_data(self):
        for r in ROWS:
            flagged = is_undertrained(DatasetFeatures(r["num_vectors"], r["dimension"], 1.0), config_from_row(r))
            self.assertEqual(flagged, bool(r["faiss_warnings"]), r["config_id"])
        self.assertTrue(any(r["faiss_warnings"] for r in ROWS))
        self.assertEqual(FAISS_MIN_POINTS_PER_CENTROID, 39)


class TestMeasuredCostModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = MeasuredCostModel().fit(ROWS)

    def test_predictions_are_sane_and_memory_is_calculated(self):
        for r in ROWS:
            p = self.model.predict_row(r)
            self.assertGreaterEqual(p.predicted_recall, 0.0)
            self.assertLessEqual(p.predicted_recall, 1.0)
            self.assertGreater(p.predicted_latency_p95_ms, 0.0)
            self.assertGreater(p.predicted_build_time_s, 0.0)
            cfg = config_from_row(r)
            self.assertEqual(p.estimated_memory_mb,
                             estimate_index_memory_mb(cfg.family, r["num_vectors"], r["dimension"], cfg))
            self.assertEqual(p.undertrained, bool(r["faiss_warnings"]))
            if r["family"] == "FLAT":
                self.assertEqual(p.predicted_recall, 1.0)

    def test_in_sample_fit_tracks_measurements(self):
        rec = [(self.model.predict_row(r).predicted_recall, r["recall_at_k"]) for r in ROWS if r["family"] != "FLAT"]
        self.assertGreater(np.corrcoef(*zip(*rec))[0, 1], 0.9)

    def test_build_model_uses_one_sample_per_build(self):
        for fam, s in self.model.training_summary["families"].items():
            builds = {r["structure_id"] for r in ROWS if r["family"] == fam}
            self.assertEqual(s["builds"], len(builds), fam)

    def test_unknown_family_and_empty_training_rejected(self):
        with self.assertRaises(ValueError):
            MeasuredCostModel().fit([])
        only_hnsw = MeasuredCostModel().fit([r for r in ROWS if r["family"] == "HNSW"])
        with self.assertRaises(ValueError):
            only_hnsw.predict(DatasetFeatures(3000, 32, 8.0), IndexConfig(family=IndexFamily.IVF_PQ, nlist=16, m_pq=8))


class TestLeaveOneDatasetOut(unittest.TestCase):

    def test_held_out_dataset_never_influences_its_own_predictions(self):
        """Corrupting every measurement of a dataset must not change that dataset's LODO predictions."""
        base = ev.lodo_predictions(ROWS)
        corrupted = [dict(r, recall_at_k=0.0, latency_p95_ms=999.0, latency_p50_ms=999.0, build_time_s=999.0)
                     if r["dataset_name"] == "tiny-clusters" else r for r in ROWS]
        after = ev.lodo_predictions(corrupted)
        for i, r in enumerate(ROWS):
            if r["dataset_name"] == "tiny-clusters":
                self.assertEqual(base[i]["recall"], after[i]["recall"])
                self.assertEqual(base[i]["p95"], after[i]["p95"])
                self.assertEqual(base[i]["build"], after[i]["build"])
        # Guard: the other folds DO train on the corrupted dataset, so their predictions must change
        changed = [i for i, r in enumerate(ROWS) if r["dataset_name"] != "tiny-clusters" and r["family"] != "FLAT"
                   and base[i]["p95"] != after[i]["p95"]]
        self.assertTrue(changed)

    def test_requires_multiple_datasets(self):
        with self.assertRaises(ValueError):
            ev.lodo_predictions([r for r in ROWS if r["dataset_name"] == "tiny-clusters"])

    def test_evaluation_report_structure_and_comparison(self):
        surrogate_before = SurrogateCostModel().latency_calibration_factor
        report = ev.evaluate(ROWS)
        for section in ("accuracy", "ranking", "selection", "memory", "undertrained"):
            self.assertIn(section, report)
        for model in ("measured_model", "synthetic_model"):
            for target in ("recall", "p95", "build"):
                self.assertIn("all", report["accuracy"][model][target])
                self.assertIn("HNSW", report["accuracy"][model][target])
        self.assertTrue(report["undertrained"]["flag_matches_recorded_warnings"])
        self.assertEqual(report["accuracy"]["measured_model"]["build"]["all"]["n"],
                         len({(r["structure_id"], r["repeat"]) for r in ROWS}))
        self.assertEqual(SurrogateCostModel().latency_calibration_factor, surrogate_before)
        self.assertIn("leave-one-dataset-out", ev.summarize(report))

    def test_cli_writes_json_report(self):
        out = os.path.join(TMP, "report.json")
        self.assertEqual(ev.main(["--data", CSV_PATH, "--output", out]), 0)
        with open(out, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["rows"], len(ROWS))

    def test_find_latest_measurements_prefers_completed_v2_runs(self):
        self.assertEqual(os.path.normpath(ev.find_latest_measurements(TMP)), os.path.normpath(CSV_PATH))
        with self.assertRaises(FileNotFoundError):
            ev.find_latest_measurements(os.path.join(TMP, "nothing-here"))


class TestMetrics(unittest.TestCase):

    def test_regression_metrics_known_values(self):
        m = ev.regression_metrics([2.0, 4.0], [1.0, 4.0], ratio_metrics=True)
        self.assertAlmostEqual(m["mae"], 0.5)
        self.assertAlmostEqual(m["rmse"], np.sqrt(0.5))
        self.assertAlmostEqual(m["r2"], 1 - 1.0 / 4.5)
        self.assertAlmostEqual(m["median_abs_pct_error"], 50.0)
        self.assertAlmostEqual(m["median_error_factor"], 1.5)
        self.assertEqual(m["within_2x"], 1.0)
        self.assertEqual(m["within_25pct"], 0.5)

    def test_rank_metrics(self):
        self.assertAlmostEqual(ev.rank_metrics([1, 2, 3, 4], [10, 20, 30, 40])["spearman"], 1.0)
        self.assertAlmostEqual(ev.rank_metrics([4, 3, 2, 1], [10, 20, 30, 40])["spearman"], -1.0)
        self.assertIsNone(ev.rank_metrics([1, 1, 1], [1, 2, 3])["spearman"])

    def test_selection_outcome(self):
        rows = [{"config_id": "A", "recall_at_k": 0.99, "latency_p95_ms": 1.0},
                {"config_id": "B", "recall_at_k": 0.96, "latency_p95_ms": 0.2},   # oracle for 0.95
                {"config_id": "C", "recall_at_k": 0.80, "latency_p95_ms": 0.1}]
        # Model thinks C reaches the target and is fastest -> direct pick misses; shortlist recovers B
        o = ev.selection_outcome(rows, pred_recall=[0.99, 0.97, 0.96], pred_p95=[1.0, 0.3, 0.05], target=0.95,
                                 shortlist_size=2)
        self.assertEqual(o["oracle"]["config_id"], "B")
        self.assertEqual(o["direct_pick"]["config_id"], "C")
        self.assertFalse(o["direct_meets_target"])
        self.assertIsNone(o["direct_latency_regret"])
        self.assertEqual(o["shortlist"], ["C", "B"])
        self.assertTrue(o["shortlist_contains_oracle"])
        self.assertEqual(o["shortlist_latency_regret"], 1.0)
        self.assertIsNone(ev.selection_outcome(rows, [1, 1, 1], [1, 1, 1], target=0.999)["oracle"])


if __name__ == "__main__":
    unittest.main()
