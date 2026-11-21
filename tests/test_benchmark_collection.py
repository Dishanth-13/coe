"""
Tests for the real-measurement benchmark collector (vecta/benchmark/collector.py).
"""

import copy
import csv
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import faiss

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vecta.config import IndexConfig, IndexFamily, MetricType
from vecta.benchmark import collector as col
from vecta.benchmark.evaluator import BenchmarkEvaluator


def smoke_spec(**overrides):
    spec = col.CollectionSpec.from_dict(col.PRESETS["smoke"].to_dict())
    for key, value in overrides.items():
        setattr(spec, key, value)
    return spec


class CollectorTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="vecta-bench-")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def run_spec(self, spec, run_id="run"):
        manifest = col.run_collection(spec, output_dir=self.tmp, run_id=run_id, progress=None)
        csv_path = os.path.join(self.tmp, run_id, col.MEASUREMENTS_FILE)
        return manifest, col.load_measurements(csv_path, include_errors=True), csv_path


class TestSmokeRun(CollectorTestBase):

    @classmethod
    def setUpClass(cls):
        cls.threads_before = faiss.omp_get_max_threads()

    def setUp(self):
        super().setUp()
        self.manifest, self.rows, self.csv_path = self.run_spec(smoke_spec())

    def test_one_row_per_planned_configuration(self):
        self.assertEqual(len(self.rows), col.count_planned_rows(smoke_spec()))
        self.assertEqual(self.manifest["rows_written"], len(self.rows))
        self.assertEqual(self.manifest["errors"], 0)
        self.assertEqual({r["family"] for r in self.rows}, {"FLAT", "HNSW", "IVF_FLAT", "IVF_PQ"})
        self.assertTrue(all(r["status"] == "ok" for r in self.rows))

    def test_csv_header_is_the_documented_schema(self):
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))
        self.assertEqual(header, col.CSV_COLUMNS)

    def test_measurements_are_physically_sane(self):
        for r in self.rows:
            self.assertGreaterEqual(r["recall_at_k"], 0.0)
            self.assertLessEqual(r["recall_at_k"], 1.0)
            self.assertLessEqual(r["latency_p50_ms"], r["latency_p95_ms"])
            self.assertLessEqual(r["latency_p95_ms"], r["latency_p99_ms"])
            self.assertGreater(r["serialized_index_mb"], 0.0)
            self.assertGreater(r["build_time_s"], 0.0)
            self.assertGreaterEqual(r["estimated_memory_mb"] + 0.01, r["serialized_index_mb"])
        flat = [r for r in self.rows if r["family"] == "FLAT"]
        self.assertEqual(len(flat), 1)
        self.assertEqual(flat[0]["recall_at_k"], 1.0)

    def test_values_are_recorded_at_full_precision(self):
        """Fast builds/queries must not be rounded away (a ~0.1 ms Flat build used to record 0.0 s)."""
        flat = next(r for r in self.rows if r["family"] == "FLAT")
        self.assertGreater(flat["build_time_s"], 0.0)
        self.assertLess(flat["build_time_s"], 0.05)
        latencies = [r[c] for r in self.rows for c in ("latency_p50_ms", "latency_p95_ms", "latency_p99_ms")]
        # Display rounding would leave every value on a 0.001 ms grid
        self.assertTrue(any(abs(v * 1000 - round(v * 1000)) > 1e-6 for v in latencies))
        recalls = {r["recall_at_k"] for r in self.rows}
        self.assertTrue(all(abs(v * 300 - round(v * 300)) < 1e-9 for v in recalls))  # exact hits / (30 q x k=10)

    def test_flat_serialized_size_matches_raw_vector_payload(self):
        flat = next(r for r in self.rows if r["family"] == "FLAT")
        payload_mb = flat["num_vectors"] * flat["dimension"] * 4 / (1024 * 1024)
        self.assertGreater(flat["serialized_index_mb"], payload_mb)
        self.assertLess(flat["serialized_index_mb"], payload_mb + 0.01)  # small FAISS header only

    def test_parameters_are_recorded_per_family_and_blank_otherwise(self):
        for r in self.rows:
            relevant = set(col.FAMILY_PARAMS[r["family"]])
            for p in col.PARAM_NAMES:
                if p in relevant:
                    self.assertIsNotNone(r[f"req_{p}"], (r["config_id"], p))
                    self.assertEqual(r[f"req_{p}"], r[f"eff_{p}"])  # no adjustments in this grid
                else:
                    self.assertIsNone(r[f"req_{p}"], (r["config_id"], p))
                    self.assertIsNone(r[f"eff_{p}"], (r["config_id"], p))
            self.assertIsNone(r["config_adjustments"])

    def test_runtime_variants_share_one_build(self):
        hnsw = [r for r in self.rows if r["family"] == "HNSW"]
        self.assertEqual(len(hnsw), 2)
        self.assertEqual(hnsw[0]["structure_id"], hnsw[1]["structure_id"])
        self.assertEqual(hnsw[0]["build_time_s"], hnsw[1]["build_time_s"])
        self.assertNotEqual(hnsw[0]["req_efSearch"], hnsw[1]["req_efSearch"])
        # More search effort must not lower recall on the same graph
        low, high = sorted(hnsw, key=lambda r: r["req_efSearch"])
        self.assertGreaterEqual(high["recall_at_k"], low["recall_at_k"])

    def test_dataset_and_hardware_metadata_recorded(self):
        r = self.rows[0]
        for key in ["dataset_fingerprint", "profile_intrinsic_dimension", "profile_hopkins",
                    "profile_norm_variance", "hardware_id"]:
            self.assertIsNotNone(r[key], key)
        with open(os.path.join(self.tmp, "run", col.MANIFEST_FILE), encoding="utf-8") as f:
            m = json.load(f)
        self.assertEqual(m["status"], "completed")
        self.assertEqual(m["spec_hash"], smoke_spec().spec_hash())
        for key in ["processor", "logical_cpus", "total_ram_gb", "faiss_version", "numpy_version",
                    "faiss_threads", "hardware_id"]:
            self.assertIn(key, m["hardware"])
        self.assertEqual(m["hardware"]["hardware_id"], r["hardware_id"])
        self.assertEqual(m["datasets"][0]["fingerprint"], r["dataset_fingerprint"])
        self.assertIn("estimated_memory_mb", m["measurement_notes"]["estimated"])
        self.assertNotIn("estimated_memory_mb", m["measurement_notes"]["measured"])

    def test_thread_setting_applied_and_restored(self):
        self.assertTrue(all(r["faiss_threads"] == 1 for r in self.rows))
        self.assertEqual(faiss.omp_get_max_threads(), self.threads_before)


class TestRepeatabilityAndEquivalence(CollectorTestBase):

    def test_two_runs_produce_identical_data_and_quality_metrics(self):
        _, a, _ = self.run_spec(smoke_spec(), run_id="a")
        _, b, _ = self.run_spec(smoke_spec(), run_id="b")
        key = lambda r: (r["dataset_name"], r["config_id"], r["repeat"])
        a, b = sorted(a, key=key), sorted(b, key=key)
        self.assertEqual([key(r) for r in a], [key(r) for r in b])
        for ra, rb in zip(a, b):
            for field in ["dataset_fingerprint", "profile_intrinsic_dimension", "profile_hopkins",
                          "recall_at_k", "serialized_index_mb"]:
                self.assertEqual(ra[field], rb[field], (ra["config_id"], field))

    def test_build_once_measurements_equal_fresh_builds(self):
        """Measuring runtime variants on a shared build must match building each config from scratch."""
        spec = smoke_spec()
        _, rows, _ = self.run_spec(spec)
        ds = spec.datasets[0]
        db, q = col.generate_dataset(ds)
        previous = faiss.omp_get_max_threads()
        faiss.omp_set_num_threads(spec.threads)
        try:
            ev = BenchmarkEvaluator(db, q, metric=MetricType(ds.metric), k=spec.k)
            for r in rows:
                if r["family"] == "FLAT":
                    continue
                cfg = IndexConfig(family=IndexFamily(r["family"]),
                                  **{p: r[f"req_{p}"] for p in col.FAMILY_PARAMS[r["family"]]})
                fresh = ev.evaluate_config(cfg)  # display-rounded (recall to 4 decimals)
                self.assertEqual(fresh.recall_at_k, round(r["recall_at_k"], 4), r["config_id"])
                self.assertAlmostEqual(fresh.index_serialized_mb, r["serialized_index_mb"], places=5)
        finally:
            faiss.omp_set_num_threads(previous)

    def test_repeats_rebuild_and_number_rows(self):
        spec = smoke_spec(repeats=2)
        _, rows, _ = self.run_spec(spec)
        self.assertEqual(len(rows), 2 * col.count_planned_rows(smoke_spec()))
        for cfg_id in {r["config_id"] for r in rows}:
            self.assertEqual(sorted(r["repeat"] for r in rows if r["config_id"] == cfg_id), [0, 1])

    def test_profiling_does_not_disturb_global_random_state(self):
        np.random.seed(123)
        expected = np.random.rand(3)
        np.random.seed(123)
        ds = smoke_spec().datasets[0]
        db, _ = col.generate_dataset(ds)
        col.profile_dataset(db, ds, k=10)
        np.testing.assert_array_equal(np.random.rand(3), expected)


class TestPlanningAndFailures(CollectorTestBase):

    def test_incompatible_configurations_are_skipped_and_reported(self):
        spec = smoke_spec(ivf_flat_nlist=[16, 4096], ivf_flat_nprobe=[1, 32], ivf_pq_m=[8, 12])
        manifest, rows, _ = self.run_spec(spec)
        skipped = " | ".join(manifest["skipped_configurations"])
        self.assertIn("nlist=4096: exceeds trainable limit 250", skipped)
        self.assertIn("nlist=16 nprobe=32: nprobe > nlist", skipped)
        self.assertIn("m_pq=12: does not divide D=32", skipped)
        ids = {r["config_id"] for r in rows}
        self.assertFalse(any("nlist=4096" in i or "M_pq=12" in i or "nprobe=32" in i for i in ids))
        self.assertTrue(all(r["config_adjustments"] is None for r in rows))

    def test_failed_build_is_recorded_and_run_continues(self):
        real_build = col.FaissIndexWrapper.build_and_add

        def flaky_build(self, vectors):
            if self.config.family == IndexFamily.HNSW:
                raise RuntimeError("simulated HNSW failure")
            return real_build(self, vectors)

        with mock.patch.object(col.FaissIndexWrapper, "build_and_add", flaky_build):
            manifest, rows, _ = self.run_spec(smoke_spec())
        errors = [r for r in rows if r["status"] == "error"]
        self.assertEqual({r["family"] for r in errors}, {"HNSW"})
        self.assertEqual(len(errors), 2)
        self.assertIn("simulated HNSW failure", errors[0]["error"])
        self.assertIsNone(errors[0]["recall_at_k"])
        self.assertEqual(manifest["errors"], 2)
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(len([r for r in rows if r["status"] == "ok"]), col.count_planned_rows(smoke_spec()) - 2)
        # load_measurements skips error rows by default
        ok_only = col.load_measurements(os.path.join(self.tmp, "run", col.MEASUREMENTS_FILE))
        self.assertTrue(all(r["status"] == "ok" for r in ok_only))

    def test_existing_results_are_never_overwritten(self):
        self.run_spec(smoke_spec())
        with self.assertRaises(FileExistsError):
            self.run_spec(smoke_spec())

    def test_spec_validation(self):
        bad = smoke_spec()
        bad.datasets[0].generator = "nope"
        with self.assertRaises(ValueError):
            bad.validate()
        dup = smoke_spec()
        dup.datasets = dup.datasets + [copy.deepcopy(dup.datasets[0])]
        with self.assertRaises(ValueError):
            dup.validate()
        with self.assertRaises(ValueError):
            smoke_spec(k=100000).validate()

    def test_spec_json_round_trip_and_cli(self):
        spec_path = os.path.join(self.tmp, "spec.json")
        self.assertEqual(col.main(["--preset", "smoke", "--write-spec", spec_path]), 0)
        with open(spec_path, encoding="utf-8") as f:
            loaded = col.CollectionSpec.from_dict(json.load(f))
        self.assertEqual(loaded.spec_hash(), col.PRESETS["smoke"].spec_hash())
        self.assertEqual(col.main(["--spec", spec_path, "--output-dir", self.tmp, "--run-id", "cli",
                                   "--threads", "1"]), 0)
        rows = col.load_measurements(os.path.join(self.tmp, "cli", col.MEASUREMENTS_FILE))
        self.assertEqual(len(rows), col.count_planned_rows(loaded))


class TestLatencyRepeatsAndWarnings(CollectorTestBase):

    def test_latency_passes_reuse_one_build_per_structure(self):
        builds, passes = [], []
        real_build = col.FaissIndexWrapper.build_and_add
        real_measure = col.BenchmarkEvaluator.measure_index

        def counting_build(self, vectors):
            builds.append(self.config.id_string)
            return real_build(self, vectors)

        def counting_measure(self, *args, **kwargs):
            passes.append(1)
            return real_measure(self, *args, **kwargs)

        spec = smoke_spec()
        with mock.patch.object(col.FaissIndexWrapper, "build_and_add", counting_build), \
                mock.patch.object(col.BenchmarkEvaluator, "measure_index", counting_measure):
            _, rows, _ = self.run_spec(spec)
        n_structures = len(col.plan_structures(spec, 2000, 32)[0])
        # One exact ground-truth index per dataset + one build per structure (not one per timing pass)
        self.assertEqual(len(builds), 1 + n_structures)
        self.assertEqual(len(passes), len(rows) * 3)            # 3 timing passes per configuration

    def test_representative_value_and_variability_columns(self):
        _, rows, _ = self.run_spec(smoke_spec())
        for r in rows:
            self.assertEqual(r["latency_repeats"], 3)
            self.assertLessEqual(r["latency_p50_ms_min"], r["latency_p50_ms"])
            self.assertLessEqual(r["latency_p50_ms"], r["latency_p50_ms_max"])
            self.assertLessEqual(r["latency_p95_ms_min"], r["latency_p95_ms"])
            self.assertLessEqual(r["latency_p95_ms"], r["latency_p95_ms_max"])
            self.assertGreaterEqual(r["latency_p95_cv"], 0.0)
            self.assertEqual(r["recall_spread"], 0.0)  # search is deterministic

    def test_summary_uses_median_of_passes(self):
        fake = [mock.Mock(latency_p50_ms=v, latency_p95_ms=v * 2, latency_p99_ms=v * 3, latency_mean_ms=v,
                          throughput_qps=1000 / v, recall_at_k=0.9) for v in (1.0, 5.0, 2.0)]
        s = col.summarize_latency_passes(fake)
        self.assertEqual(s["latency_p50_ms"], 2.0)
        self.assertEqual(s["latency_p95_ms"], 4.0)
        self.assertEqual((s["latency_p95_ms_min"], s["latency_p95_ms_max"]), (2.0, 10.0))
        self.assertAlmostEqual(s["latency_p95_cv"], float(np.std([2, 10, 4]) / np.mean([2, 10, 4])))
        self.assertEqual(s["recall_spread"], 0.0)

    def test_faiss_training_warnings_are_recorded(self):
        # The smoke PQ codebook (256 centroids) is trained on only 2000 vectors -> FAISS warns
        manifest, rows, _ = self.run_spec(smoke_spec())
        pq = [r for r in rows if r["family"] == "IVF_PQ"]
        self.assertTrue(pq and all("WARNING clustering 2000 points to 256 centroids" in r["faiss_warnings"] for r in pq))
        for r in rows:
            if r["family"] in ("FLAT", "HNSW"):
                self.assertIsNone(r["faiss_warnings"], r["config_id"])
        entries = {e["structure_id"]: e for e in manifest["faiss_warnings"]}
        self.assertIn(pq[0]["structure_id"], entries)
        self.assertEqual(len(entries[pq[0]["structure_id"]]["warnings"]),
                         len(set(entries[pq[0]["structure_id"]]["warnings"])))  # de-duplicated

    def test_native_stderr_capture_restores_stderr(self):
        x = np.random.RandomState(0).randn(500, 8).astype("float32")
        with col.capture_native_stderr() as cap:
            q = faiss.IndexFlatL2(8)
            idx = faiss.IndexIVFFlat(q, 8, 50)
            idx.train(x)
        self.assertIn("WARNING clustering 500 points to 50 centroids", cap.text)
        # fd 2 is usable again after the block
        os.write(2, b"")
        self.assertEqual(col.parse_faiss_warnings("a\n\n a \nb\na\n"), ["a", "b"])

    def test_schema_v1_columns_preserved_and_v1_files_still_load(self):
        v1_end = col.CSV_COLUMNS.index("hardware_id") + 1
        self.assertEqual(v1_end, 49)
        appended = col.CSV_COLUMNS[v1_end:]
        self.assertEqual(appended[0], "faiss_warnings")
        _, _, csv_path = self.run_spec(smoke_spec())
        v1_path = os.path.join(self.tmp, "v1.csv")
        with open(csv_path, newline="", encoding="utf-8") as src, open(v1_path, "w", newline="", encoding="utf-8") as dst:
            reader = csv.DictReader(src)
            writer = csv.DictWriter(dst, fieldnames=col.CSV_COLUMNS[:v1_end], extrasaction="ignore")
            writer.writeheader()
            for row in reader:
                writer.writerow(row)
        old = col.load_measurements(v1_path)
        self.assertEqual(len(old), col.count_planned_rows(smoke_spec()))
        self.assertNotIn("latency_p95_cv", old[0])

    def test_older_spec_json_without_latency_repeats_loads(self):
        data = col.PRESETS["smoke"].to_dict()
        data.pop("latency_repeats")
        self.assertEqual(col.CollectionSpec.from_dict(data).latency_repeats, 3)


class TestDefaultPresetScope(unittest.TestCase):

    def test_default_grid_matches_requested_scope(self):
        spec = col.PRESETS["default"]
        spec.validate()
        self.assertEqual(len(spec.datasets), 5)
        self.assertEqual({d.generator for d in spec.datasets}, set(col.GENERATORS))
        self.assertEqual(len({(d.num_vectors, d.dimension, d.generator) for d in spec.datasets}), 5)
        # High-recall region is covered
        self.assertTrue({256, 512} <= set(spec.hnsw_ef_search))
        self.assertGreaterEqual(max(spec.ivf_flat_nprobe), 256)
        self.assertEqual(spec.latency_repeats, 3)
        for ds in spec.datasets:
            plan, skipped = col.plan_structures(spec, ds.num_vectors, ds.dimension)
            # Only invalid nprobe > nlist combinations may be skipped
            self.assertTrue(all("nprobe > nlist" in s for s in skipped), skipped)
            nprobes = {(s.build_config.nlist, v.nprobe) for s in plan if s.family == IndexFamily.IVF_FLAT
                       for v in s.variants}
            self.assertIn((1024, 256), nprobes)
            self.assertTrue(all(nprobe <= nlist for nlist, nprobe in nprobes))
            per_family = {}
            for s in plan:
                per_family[s.family.value] = per_family.get(s.family.value, 0) + len(s.variants)
            self.assertEqual(per_family["FLAT"], 1)
            for fam in ("HNSW", "IVF_FLAT", "IVF_PQ"):
                self.assertGreaterEqual(per_family[fam], 10, fam)
                self.assertLessEqual(per_family[fam], 20, fam)
        self.assertEqual(spec.threads, 1)


if __name__ == "__main__":
    unittest.main()
