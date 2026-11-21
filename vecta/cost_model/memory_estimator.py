"""
VECTA Index-Specific Memory Estimator.

Deterministic (non-ML) estimate of a FAISS index's size from its structure, following what each
index type stores. It is validated against the MEASURED serialized index size collected by the
benchmark pipeline; it estimates index data size, not total process RAM.
"""

from typing import Optional
from vecta.config import IndexConfig, IndexFamily

MB = 1024.0 * 1024.0
FLOAT_BYTES = 4
ID_BYTES = 8          # faiss.idx_t stored per vector in inverted lists
INT_BYTES = 4         # HNSW neighbour ids / levels
OFFSET_BYTES = 8      # HNSW per-node offsets (size_t)


def hnsw_neighbor_slots_per_node(M: int) -> float:
    """
    Expected neighbour slots per node in faiss::HNSW: level 0 has 2*M slots and every upper level M.
    FAISS draws levels with P(level >= l) = M^-l, so the expected number of upper levels is 1/(M-1).
    """
    M = max(2, int(M))
    return 2.0 * M + M / (M - 1.0)


def estimate_index_bytes(family: IndexFamily, num_vectors: int, dimension: int,
                         config: Optional[IndexConfig] = None) -> float:
    """Estimated size in bytes of the index data structures (vectors/codes, graph, centroids, ids)."""
    n, d = int(num_vectors), int(dimension)
    vectors = n * d * FLOAT_BYTES
    if family == IndexFamily.FLAT:
        return float(vectors)
    if config is None:
        raise ValueError(f"{family.value} needs an IndexConfig")
    if family == IndexFamily.HNSW:
        graph = n * hnsw_neighbor_slots_per_node(config.M) * INT_BYTES
        per_node_meta = n * (OFFSET_BYTES + INT_BYTES)  # offsets + level of each node
        return float(vectors + graph + per_node_meta)
    if family == IndexFamily.IVF_FLAT:
        centroids = config.nlist * d * FLOAT_BYTES
        list_sizes = config.nlist * ID_BYTES
        return float(vectors + n * ID_BYTES + centroids + list_sizes)
    if family == IndexFamily.IVF_PQ:
        code_bytes = n * config.m_pq * config.nbits_per_idx / 8.0
        centroids = config.nlist * d * FLOAT_BYTES
        codebooks = (1 << config.nbits_per_idx) * d * FLOAT_BYTES  # m_pq x 2^nbits x (d / m_pq) floats
        list_sizes = config.nlist * ID_BYTES
        return float(code_bytes + n * ID_BYTES + centroids + codebooks + list_sizes)
    raise ValueError(f"unsupported family {family}")


def estimate_index_memory_mb(family: IndexFamily, num_vectors: int, dimension: int,
                             config: Optional[IndexConfig] = None) -> float:
    return estimate_index_bytes(family, num_vectors, dimension, config) / MB
