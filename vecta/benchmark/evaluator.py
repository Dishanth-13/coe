"""
VECTA Empirical Benchmark Evaluator.
Computes exact ground-truth nearest neighbors, Recall@k accuracy,
fine-grained latency percentiles (p50, p95, p99), serialized index size, estimated index
memory, and sequential single-query throughput (QPS).
"""

import time
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from vecta.config import IndexConfig, BenchmarkResult, MetricType, IndexFamily
from vecta.engine.faiss_wrapper import FaissIndexWrapper


class BenchmarkEvaluator:
    """Rigorous evaluation suite validating ANN performance against exact brute-force ground truth."""

    def __init__(
        self,
        database: np.ndarray,
        queries: np.ndarray,
        metric: MetricType = MetricType.L2,
        k: int = 10
    ):
        self.database = np.ascontiguousarray(database, dtype=np.float32)
        self.queries = np.ascontiguousarray(queries, dtype=np.float32)
        self.metric = metric
        self.k = k
        self.dimension = database.shape[1]
        
        # Ground truth structures
        self.gt_indices: Optional[np.ndarray] = None
        self.gt_distances: Optional[np.ndarray] = None
        self.gt_query_latency_ms: float = 0.0
        
        self._compute_ground_truth()

    def _compute_ground_truth(self):
        """Compute exact nearest neighbors using brute-force IndexFlat."""
        flat_cfg = IndexConfig(family=IndexFamily.FLAT)
        flat_wrapper = FaissIndexWrapper(flat_cfg, self.dimension, self.metric)
        flat_wrapper.build_and_add(self.database)
        self._flat_wrapper = flat_wrapper

        dists, indices, _ = flat_wrapper.search(self.queries, k=self.k)
        self.gt_indices = indices
        self.gt_distances = dists

        # Mean single-query exact-search latency, measured the same way as evaluate_config()
        # so that speedup_vs_baseline compares like with like.
        single_latencies = []
        for q_idx in range(len(self.queries)):
            t0 = time.perf_counter()
            flat_wrapper.search(self.queries[q_idx:q_idx + 1], k=self.k)
            single_latencies.append((time.perf_counter() - t0) * 1000.0)
        self.gt_query_latency_ms = float(np.mean(single_latencies)) if single_latencies else 0.0

    def exact_search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Exact brute-force k-NN over this evaluator's database (same metric rules as the GT)."""
        dists, indices, _ = self._flat_wrapper.search(queries, k=k)
        return dists, indices

    @staticmethod
    def calculate_recall_at_k(retrieved_indices: np.ndarray, ground_truth_indices: np.ndarray, k: int) -> float:
        """
        Compute mean Recall@k across all query instances:
        Recall@k = (1 / Q) * sum_q (|retrieved_q[:k] intersect gt_q[:k]| / k)
        """
        assert retrieved_indices.shape[0] == ground_truth_indices.shape[0]
        q_count = retrieved_indices.shape[0]
        
        total_intersection = 0
        for i in range(q_count):
            ret_set = set(retrieved_indices[i, :k])
            gt_set = set(ground_truth_indices[i, :k])
            total_intersection += len(ret_set.intersection(gt_set))
            
        return float(total_intersection / (q_count * k))

    def evaluate_config(self, config: IndexConfig, warmup_runs: int = 2) -> BenchmarkResult:
        """Construct index, run queries with per-query timing, and compute performance metrics."""
        wrapper = FaissIndexWrapper(config, self.dimension, self.metric)
        build_time = wrapper.build_and_add(self.database)
        return self.measure_index(wrapper, build_time, warmup_runs=warmup_runs)

    def measure_index(self, wrapper: FaissIndexWrapper, build_time: float, warmup_runs: int = 2,
                      round_results: bool = True) -> BenchmarkResult:
        """
        Measure an index that is already built on this evaluator's database, using its *current*
        search-time parameters (efSearch / nprobe). Lets callers build a structure once and measure
        several runtime settings; `build_time` is reported as given (the shared build's time).
        `round_results=False` keeps full precision (for recorded benchmark data); the default rounds
        for display, as before.
        """
        # Warmup
        for _ in range(warmup_runs):
            wrapper.search(self.queries[:min(5, len(self.queries))], k=self.k)
            
        # Individual query timings for robust percentile analysis
        query_latencies = []
        retrieved_all = []
        
        # Evaluate queries individually to capture micro-jitter and tail latencies
        for q_idx in range(len(self.queries)):
            single_q = self.queries[q_idx:q_idx + 1]
            t0 = time.perf_counter()
            dists, idxs, _ = wrapper.search(single_q, k=self.k)
            t_ms = (time.perf_counter() - t0) * 1000.0
            
            query_latencies.append(t_ms)
            retrieved_all.append(idxs[0])
            
        retrieved_matrix = np.array(retrieved_all)
        lat_arr = np.array(query_latencies)
        
        # Metric extraction
        recall = self.calculate_recall_at_k(retrieved_matrix, self.gt_indices, self.k)
        p50 = float(np.percentile(lat_arr, 50))
        p95 = float(np.percentile(lat_arr, 95))
        p99 = float(np.percentile(lat_arr, 99))
        mean_lat = float(np.mean(lat_arr))
        
        total_query_time_s = np.sum(lat_arr) / 1000.0
        qps = float(len(self.queries) / max(1e-6, total_query_time_s))
        
        speedup = self.gt_query_latency_ms / max(1e-4, mean_lat)
        rnd = (lambda x, digits: round(x, digits)) if round_results else (lambda x, digits: float(x))

        return BenchmarkResult(
            config=wrapper.config,  # effective configuration actually built by FAISS
            recall_at_k=rnd(recall, 4),
            latency_p50_ms=rnd(p50, 3),
            latency_p95_ms=rnd(p95, 3),
            latency_p99_ms=rnd(p99, 3),
            latency_mean_ms=rnd(mean_lat, 3),
            throughput_qps=rnd(qps, 1),
            memory_footprint_mb=rnd(wrapper.estimated_memory_mb, 2),
            index_build_time_s=rnd(build_time, 3),
            speedup_vs_baseline=rnd(speedup, 2),
            requested_config=wrapper.requested_config,
            config_adjustments=list(wrapper.config_adjustments),
            index_serialized_mb=float(wrapper.serialized_size_mb)
        )
