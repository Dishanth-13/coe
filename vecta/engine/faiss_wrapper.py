"""
VECTA High-Performance FAISS Engine Wrapper.
Unifies IndexFlat, IndexHNSWFlat, IndexIVFFlat, and IndexIVFPQ with precise telemetry,
index-size estimation, and ground-truth generation.
"""

import time
import tempfile
import os
import numpy as np
import faiss
from dataclasses import replace
from typing import Tuple, Optional, Dict, Any, List
from vecta.config import (
    IndexConfig,
    IndexFamily,
    MetricType,
    MIN_TRAINING_VECTORS_PER_CENTROID,
    max_ivf_nlist,
    largest_valid_pq_m,
    max_pq_nbits
)


def l2_normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """
    Return a new float32 array with each row scaled to unit L2 norm.
    Zero rows are left as zeros (their cosine similarity to anything is defined as 0).
    The input array is never modified.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


class FaissIndexWrapper:
    """Encapsulates FAISS index construction, vector indexing, query execution, and serialization."""

    def __init__(self, config: IndexConfig, dimension: int, metric: MetricType = MetricType.L2):
        # Work on private copies so building / runtime tuning never mutates the caller's object
        # (e.g. the shared FAISS_DEFAULT_* constants). After build_and_add(), `self.config` is the
        # *effective* configuration FAISS actually built; `requested_config` is what was asked for.
        self.requested_config = replace(config)
        self.config = replace(config)
        self.config_adjustments: List[str] = []
        self.dimension = dimension
        self.metric = metric
        # COSINE is implemented as inner product over L2-normalized database and query vectors
        self.faiss_metric = faiss.METRIC_L2 if metric == MetricType.L2 else faiss.METRIC_INNER_PRODUCT
        self.index: Optional[faiss.Index] = None
        self.is_trained: bool = False
        self.total_vectors: int = 0
        self.build_time_s: float = 0.0
        # Estimated in-memory index size (see _compute_memory_footprint_mb); NOT measured RAM
        self.estimated_memory_mb: float = 0.0
        # Measured size of the serialized FAISS index (bytes written by faiss.write_index), in MB
        self.serialized_size_mb: float = 0.0

    def build_and_add(self, vectors: np.ndarray) -> float:
        """Instantiate index, train if necessary, add vectors, and record build time."""
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if self.metric == MetricType.COSINE:
            vectors = l2_normalize_rows(vectors)
        n, d = vectors.shape
        assert d == self.dimension, f"Vector dimension {d} does not match configured {self.dimension}"

        start_time = time.perf_counter()

        if self.config.family == IndexFamily.FLAT:
            if self.metric == MetricType.L2:
                self.index = faiss.IndexFlatL2(d)
            else:
                self.index = faiss.IndexFlatIP(d)
            self.index.add(vectors)
            self.is_trained = True

        elif self.config.family == IndexFamily.HNSW:
            self.index = faiss.IndexHNSWFlat(d, self.config.M, self.faiss_metric)
            self.index.hnsw.efConstruction = int(self.config.efConstruction)
            self.index.hnsw.efSearch = int(self.config.efSearch)
            self.index.add(vectors)
            self.is_trained = True

        elif self.config.family == IndexFamily.IVF_FLAT:
            quantizer = faiss.IndexFlatL2(d) if self.metric == MetricType.L2 else faiss.IndexFlatIP(d)
            self._resolve_ivf_parameters(n)
            self.index = faiss.IndexIVFFlat(quantizer, d, int(self.config.nlist), self.faiss_metric)
            self.index.train(vectors)
            self.index.add(vectors)
            self.index.nprobe = int(self.config.nprobe)
            self.is_trained = True

        elif self.config.family == IndexFamily.IVF_PQ:
            quantizer = faiss.IndexFlatL2(d) if self.metric == MetricType.L2 else faiss.IndexFlatIP(d)
            self._resolve_ivf_parameters(n)
            self._resolve_pq_parameters(n, d)
            self.index = faiss.IndexIVFPQ(
                quantizer, d, int(self.config.nlist), int(self.config.m_pq),
                int(self.config.nbits_per_idx), self.faiss_metric
            )
            self.index.train(vectors)
            self.index.add(vectors)
            self.index.nprobe = int(self.config.nprobe)
            self.is_trained = True

        else:
            raise ValueError(f"Unsupported IndexFamily: {self.config.family}")

        self.build_time_s = time.perf_counter() - start_time
        self.total_vectors = self.index.ntotal
        self.estimated_memory_mb = self._compute_memory_footprint_mb(n, d)
        return self.build_time_s

    def _adjust(self, name: str, new_value: int, reason: str):
        """Apply a FAISS-compatibility change to the effective config and record it."""
        old_value = getattr(self.config, name)
        if int(new_value) != int(old_value):
            setattr(self.config, name, int(new_value))
            self.config_adjustments.append(f"{name}: {old_value} -> {int(new_value)} ({reason})")

    def _resolve_ivf_parameters(self, n: int):
        """Clamp nlist to the training-set size and nprobe to nlist."""
        cap = max_ivf_nlist(n)
        if self.config.nlist > cap:
            self._adjust("nlist", cap, f"needs >= {MIN_TRAINING_VECTORS_PER_CENTROID} training vectors per "
                                       f"centroid; dataset has {n} vectors")
        if self.config.nprobe > self.config.nlist:
            self._adjust("nprobe", self.config.nlist, "cannot probe more lists than nlist")
        if self.config.nprobe < 1:
            self._adjust("nprobe", 1, "nprobe must be >= 1")

    def _resolve_pq_parameters(self, n: int, d: int):
        """Make m_pq divide the dimension and keep the PQ codebook trainable."""
        valid_m = largest_valid_pq_m(d, self.config.m_pq)
        if valid_m != self.config.m_pq:
            self._adjust("m_pq", valid_m, f"must divide dimension {d}; using the largest divisor <= requested")
        valid_nbits = max_pq_nbits(n, self.config.nbits_per_idx)
        if valid_nbits != self.config.nbits_per_idx:
            self._adjust("nbits_per_idx", valid_nbits, f"2^nbits codebook centroids need >= that many "
                                                       f"training vectors; dataset has {n}")

    def search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Execute batch ANN search.
        Returns: (distances, indices, query_latency_ms)
        """
        if self.index is None:
            raise RuntimeError("Index has not been constructed yet.")

        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if self.metric == MetricType.COSINE:
            queries = l2_normalize_rows(queries)

        # Apply runtime query parameter settings if supported
        if self.config.family == IndexFamily.HNSW:
            self.index.hnsw.efSearch = int(self.config.efSearch)
        elif self.config.family in (IndexFamily.IVF_FLAT, IndexFamily.IVF_PQ):
            self.index.nprobe = int(self.config.nprobe)

        t0 = time.perf_counter()
        distances, indices = self.index.search(queries, k)
        t_elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return distances, indices, t_elapsed_ms

    def set_runtime_parameters(self, efSearch: Optional[int] = None, nprobe: Optional[int] = None):
        """Dynamic runtime parameter adjustment without index rebuild."""
        if efSearch is not None and self.config.family == IndexFamily.HNSW and self.index is not None:
            self.config.efSearch = int(efSearch)
            self.index.hnsw.efSearch = int(efSearch)
        if nprobe is not None and self.config.family in (IndexFamily.IVF_FLAT, IndexFamily.IVF_PQ) and self.index is not None:
            effective = max(1, min(int(nprobe), int(self.config.nlist)))
            if effective != int(nprobe):
                self.config_adjustments.append(
                    f"nprobe: requested {int(nprobe)} at runtime -> {effective} (must be within 1..nlist={self.config.nlist})"
                )
            self.config.nprobe = effective
            self.index.nprobe = effective

    def _compute_memory_footprint_mb(self, n: int, d: int) -> float:
        """
        ESTIMATE of the index's in-memory size (MB): the larger of
          (1) the measured size of the serialized FAISS index (stored in `serialized_size_mb`), and
          (2) an analytical model of vectors / graph links / centroids / codes.
        This is not a measurement of process RAM (RSS); FAISS runtime overheads are not captured.
        """
        if self.index is None:
            return 0.0

        # Method 1: Serialized disk footprint (an actual measurement of index bytes)
        temp_file = None
        try:
            fd, temp_file = tempfile.mkstemp(suffix=".faiss")
            os.close(fd)
            faiss.write_index(self.index, temp_file)
            file_bytes = os.path.getsize(temp_file)
            serialized_mb = file_bytes / (1024 * 1024)
        except Exception:
            serialized_mb = 0.0
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass
        self.serialized_size_mb = serialized_mb

        # Method 2: Analytical model
        # Raw float32 vectors = N * D * 4 bytes
        raw_vec_mb = (n * d * 4.0) / (1024 * 1024)
        
        if self.config.family == IndexFamily.FLAT:
            analytical_mb = raw_vec_mb
        elif self.config.family == IndexFamily.HNSW:
            # HNSW overhead: node connections = N * M * 2 * 4 bytes + raw vector table
            graph_overhead_mb = (n * self.config.M * 8.0) / (1024 * 1024)
            analytical_mb = raw_vec_mb + graph_overhead_mb
        elif self.config.family == IndexFamily.IVF_FLAT:
            # Centroid table + inverted list pointer metadata
            centroid_mb = (self.config.nlist * d * 4.0) / (1024 * 1024)
            analytical_mb = raw_vec_mb + centroid_mb + (n * 8.0) / (1024 * 1024)
        elif self.config.family == IndexFamily.IVF_PQ:
            # Quantized codes (m_pq * nbits bits per vector) + PQ codebook + IVF centroids + ids
            # (all from the *effective* configuration that FAISS actually built)
            nbits = int(self.config.nbits_per_idx)
            code_mb = (n * self.config.m_pq * nbits / 8.0) / (1024 * 1024)
            codebook_mb = ((1 << nbits) * d * 4.0) / (1024 * 1024)
            centroid_mb = (self.config.nlist * d * 4.0) / (1024 * 1024)
            analytical_mb = code_mb + codebook_mb + centroid_mb + (n * 8.0) / (1024 * 1024)
        else:
            analytical_mb = raw_vec_mb

        if serialized_mb > 0:
            return round(max(serialized_mb, analytical_mb), 2)
        return round(analytical_mb, 2)
