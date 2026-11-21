"""
VECTA Real-Measurement Benchmark Collector.

Runs a controlled, configurable grid of FAISS index configurations (Flat, HNSW, IVF-Flat,
IVF-PQ) over a small set of locally generated datasets and records MEASURED results:
Recall@k, single-query latency percentiles, index build time and serialized index size,
together with dataset characteristics, requested/effective index parameters and hardware
metadata. Output is a CSV of measurements plus a JSON run manifest, intended as training and
validation data for a cost model.

Run:  python -m vecta.benchmark.collector --preset default
"""

import argparse
import contextlib
import csv
import datetime as _dt
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import faiss

from vecta.config import IndexConfig, IndexFamily, MetricType, max_ivf_nlist
from vecta.engine.faiss_wrapper import FaissIndexWrapper
from vecta.benchmark.dataset_generator import BenchmarkDatasetGenerator
from vecta.benchmark.evaluator import BenchmarkEvaluator
from vecta.profiler.workload_profiler import WorkloadProfiler
from vecta.profiler.hardware_profiler import HardwareProfiler

SCHEMA_VERSION = 2  # v2 appended latency-variability and FAISS-warning columns
DEFAULT_OUTPUT_DIR = os.path.join("benchmarks", "results")
MEASUREMENTS_FILE = "measurements.csv"
MANIFEST_FILE = "run_metadata.json"

GENERATORS: Dict[str, Callable[..., Tuple[np.ndarray, np.ndarray]]] = {
    "gaussian_clusters": BenchmarkDatasetGenerator.generate_gaussian_clusters,
    "semantic_embeddings": BenchmarkDatasetGenerator.generate_semantic_embeddings,
    "isotropic_gaussian": BenchmarkDatasetGenerator.generate_isotropic_gaussian,
}

PARAM_NAMES = ["M", "efConstruction", "efSearch", "nlist", "nprobe", "m_pq", "nbits_per_idx"]
# Parameters that are meaningful for each family (others are left blank in the CSV, never 0)
FAMILY_PARAMS = {
    "FLAT": [],
    "HNSW": ["M", "efConstruction", "efSearch"],
    "IVF_FLAT": ["nlist", "nprobe"],
    "IVF_PQ": ["nlist", "nprobe", "m_pq", "nbits_per_idx"],
}

CSV_COLUMNS = (
    ["schema_version", "run_id", "row_id", "status", "error",
     "dataset_name", "dataset_generator", "dataset_fingerprint", "dataset_seed",
     "num_vectors", "dimension", "num_queries", "metric", "k",
     "profile_intrinsic_dimension", "profile_hopkins", "profile_norm_mean", "profile_norm_variance",
     "profile_sparsity_ratio",
     "family", "config_id", "structure_id", "repeat"]
    + [f"req_{p}" for p in PARAM_NAMES]
    + [f"eff_{p}" for p in PARAM_NAMES]
    + ["config_adjustments",
       # --- measured on this host ---
       "recall_at_k", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "latency_mean_ms",
       "throughput_qps_sequential", "build_time_s", "serialized_index_mb",
       # --- estimate (max of serialized size and an analytical model), NOT measured RAM ---
       "estimated_memory_mb",
       "faiss_threads", "hardware_id",
       # --- appended in schema v2 (existing columns unchanged) ---
       "faiss_warnings",
       "latency_repeats", "latency_p50_ms_min", "latency_p50_ms_max",
       "latency_p95_ms_min", "latency_p95_ms_max", "latency_p95_cv", "recall_spread"]
)

INT_COLUMNS = {"schema_version", "row_id", "dataset_seed", "num_vectors", "dimension", "num_queries", "k",
               "repeat", "faiss_threads", "latency_repeats"} | {f"{p}_{n}" for p in ("req", "eff") for n in PARAM_NAMES}
FLOAT_COLUMNS = {"profile_intrinsic_dimension", "profile_hopkins", "profile_norm_mean", "profile_norm_variance",
                 "profile_sparsity_ratio", "recall_at_k", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
                 "latency_mean_ms", "throughput_qps_sequential", "build_time_s", "serialized_index_mb",
                 "estimated_memory_mb", "latency_p50_ms_min", "latency_p50_ms_max", "latency_p95_ms_min",
                 "latency_p95_ms_max", "latency_p95_cv", "recall_spread"}


# ---------------------------------------------------------------------------
# Specification
# ---------------------------------------------------------------------------

@dataclass
class DatasetSpec:
    """A locally generated dataset (deterministic for a given generator, size and seed)."""
    name: str
    generator: str
    num_vectors: int
    dimension: int
    num_queries: int = 200
    seed: int = 42
    metric: str = "L2"
    params: Dict[str, Any] = field(default_factory=dict)  # generator-specific keyword arguments


@dataclass
class CollectionSpec:
    """Benchmark grid. Structural parameters trigger a build; runtime parameters reuse it."""
    name: str
    datasets: List[DatasetSpec]
    include_flat: bool = True
    hnsw_builds: List[List[int]] = field(default_factory=list)      # [[M, efConstruction], ...]
    hnsw_ef_search: List[int] = field(default_factory=list)
    ivf_flat_nlist: List[int] = field(default_factory=list)
    ivf_flat_nprobe: List[int] = field(default_factory=list)
    ivf_pq_nlist: List[int] = field(default_factory=list)
    ivf_pq_m: List[int] = field(default_factory=list)
    ivf_pq_nprobe: List[int] = field(default_factory=list)
    ivf_pq_nbits: int = 8
    k: int = 10
    repeats: int = 1           # independent index builds per structure
    latency_repeats: int = 3   # timing passes per configuration on the same built index (no rebuild)
    warmup_runs: int = 2
    threads: int = 1  # FAISS/OpenMP threads, pinned for the whole run to keep latency comparable

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "CollectionSpec":
        data = dict(data)
        data["datasets"] = [DatasetSpec(**d) for d in data["datasets"]]
        return CollectionSpec(**data)

    def spec_hash(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def validate(self):
        if not self.datasets:
            raise ValueError("spec must contain at least one dataset")
        names = [d.name for d in self.datasets]
        if len(set(names)) != len(names):
            raise ValueError("dataset names must be unique")
        for d in self.datasets:
            if d.generator not in GENERATORS:
                raise ValueError(f"unknown generator {d.generator!r}; choose from {sorted(GENERATORS)}")
            if d.metric not in {m.value for m in MetricType}:
                raise ValueError(f"unknown metric {d.metric!r}")
            if d.num_vectors < 1 or d.dimension < 1 or d.num_queries < 1:
                raise ValueError(f"dataset {d.name!r} must have positive sizes")
            if self.k > d.num_vectors:
                raise ValueError(f"k={self.k} exceeds dataset {d.name!r} size")
        if self.k < 1 or self.repeats < 1 or self.threads < 1 or self.latency_repeats < 1 or self.warmup_runs < 0:
            raise ValueError("k, repeats, latency_repeats and threads must be >= 1; warmup_runs >= 0")
        for pair in self.hnsw_builds:
            if len(pair) != 2:
                raise ValueError("hnsw_builds entries must be [M, efConstruction]")


PRESETS: Dict[str, CollectionSpec] = {
    # 55 configurations per dataset: Flat 1, HNSW 20, IVF-Flat 16, IVF-PQ 18 (5 datasets = 275 rows).
    # The grid reaches into the high-recall region (efSearch up to 512, nprobe up to 256).
    "default": CollectionSpec(
        name="default",
        datasets=[
            DatasetSpec("clusters-20k-d128", "gaussian_clusters", 20000, 128, 200, seed=101,
                        params={"num_clusters": 32, "cluster_std": 0.25}),
            DatasetSpec("semantic-20k-d64", "semantic_embeddings", 20000, 64, 200, seed=202,
                        params={"intrinsic_dim": 12}),
            DatasetSpec("isotropic-20k-d96", "isotropic_gaussian", 20000, 96, 200, seed=303),
            # Higher dimension, more and tighter clusters
            DatasetSpec("clusters-20k-d256", "gaussian_clusters", 20000, 256, 200, seed=404,
                        params={"num_clusters": 64, "cluster_std": 0.15}),
            # Larger N, higher intrinsic dimension than semantic-20k-d64
            DatasetSpec("semantic-30k-d128", "semantic_embeddings", 30000, 128, 200, seed=505,
                        params={"intrinsic_dim": 32}),
        ],
        hnsw_builds=[[8, 64], [16, 64], [32, 64], [16, 200]],
        hnsw_ef_search=[16, 64, 128, 256, 512],
        ivf_flat_nlist=[64, 256, 1024],
        ivf_flat_nprobe=[1, 4, 16, 64, 128, 256],
        ivf_pq_nlist=[64, 256],
        ivf_pq_m=[8, 16, 32],
        ivf_pq_nprobe=[1, 16, 64],
    ),
    # Seconds-scale run for checking the pipeline end to end
    "smoke": CollectionSpec(
        name="smoke",
        datasets=[
            DatasetSpec("smoke-clusters-2k-d32", "gaussian_clusters", 2000, 32, 30, seed=7,
                        params={"num_clusters": 8}),
        ],
        hnsw_builds=[[8, 32]],
        hnsw_ef_search=[16, 64],
        ivf_flat_nlist=[16],
        ivf_flat_nprobe=[1, 4],
        ivf_pq_nlist=[16],
        ivf_pq_m=[8],
        ivf_pq_nprobe=[2],
    ),
}


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

@dataclass
class PlannedStructure:
    """One index build plus the runtime-parameter variants measured on it."""
    family: IndexFamily
    build_config: IndexConfig
    variants: List[IndexConfig]


def plan_structures(spec: CollectionSpec, num_vectors: int, dimension: int) -> Tuple[List[PlannedStructure], List[str]]:
    """
    Expand the grid for one dataset. Configurations FAISS would have to alter (nlist above the
    trainable limit, m_pq not dividing D, nprobe > nlist) are skipped and reported, so the data
    never contains silently adjusted duplicates.
    """
    plan: List[PlannedStructure] = []
    skipped: List[str] = []

    if spec.include_flat:
        flat = IndexConfig(family=IndexFamily.FLAT)
        plan.append(PlannedStructure(IndexFamily.FLAT, flat, [flat]))

    for M, efc in spec.hnsw_builds:
        variants = [IndexConfig(family=IndexFamily.HNSW, M=int(M), efConstruction=int(efc), efSearch=int(efs))
                    for efs in spec.hnsw_ef_search]
        if variants:
            plan.append(PlannedStructure(IndexFamily.HNSW, variants[0], variants))

    cap = max_ivf_nlist(num_vectors)
    for nlist in spec.ivf_flat_nlist:
        if nlist > cap:
            skipped.append(f"IVF_FLAT nlist={nlist}: exceeds trainable limit {cap} for N={num_vectors}")
            continue
        variants = []
        for nprobe in spec.ivf_flat_nprobe:
            if nprobe > nlist:
                skipped.append(f"IVF_FLAT nlist={nlist} nprobe={nprobe}: nprobe > nlist")
                continue
            variants.append(IndexConfig(family=IndexFamily.IVF_FLAT, nlist=int(nlist), nprobe=int(nprobe)))
        if variants:
            plan.append(PlannedStructure(IndexFamily.IVF_FLAT, variants[0], variants))

    for nlist in spec.ivf_pq_nlist:
        if nlist > cap:
            skipped.append(f"IVF_PQ nlist={nlist}: exceeds trainable limit {cap} for N={num_vectors}")
            continue
        for m in spec.ivf_pq_m:
            if dimension % m != 0:
                skipped.append(f"IVF_PQ m_pq={m}: does not divide D={dimension}")
                continue
            if (1 << spec.ivf_pq_nbits) > num_vectors:
                skipped.append(f"IVF_PQ nbits={spec.ivf_pq_nbits}: needs >= {1 << spec.ivf_pq_nbits} training vectors")
                continue
            variants = []
            for nprobe in spec.ivf_pq_nprobe:
                if nprobe > nlist:
                    skipped.append(f"IVF_PQ nlist={nlist} nprobe={nprobe}: nprobe > nlist")
                    continue
                variants.append(IndexConfig(family=IndexFamily.IVF_PQ, nlist=int(nlist), nprobe=int(nprobe),
                                            m_pq=int(m), nbits_per_idx=int(spec.ivf_pq_nbits)))
            if variants:
                plan.append(PlannedStructure(IndexFamily.IVF_PQ, variants[0], variants))
    return plan, skipped


def count_planned_rows(spec: CollectionSpec) -> int:
    total = 0
    for d in spec.datasets:
        plan, _ = plan_structures(spec, d.num_vectors, d.dimension)
        total += sum(len(s.variants) for s in plan) * spec.repeats
    return total


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def collect_hardware_metadata(threads: int) -> Dict[str, Any]:
    hw = HardwareProfiler.get_hardware_profile()
    try:
        compile_options = faiss.get_compile_options().strip()
    except Exception:
        compile_options = ""
    import sklearn
    meta = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "machine": platform.machine(),
        "logical_cpus": hw["logical_cpus"],
        "physical_cpus": hw["physical_cpus"],
        "total_ram_gb": hw["total_ram_gb"],
        "available_ram_gb_at_start": hw["available_ram_gb"],
        "python_version": platform.python_version(),
        "faiss_version": getattr(faiss, "__version__", "unknown"),
        "faiss_compile_options": compile_options,
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "faiss_threads": threads,
    }
    identity = "|".join(str(meta[k]) for k in ("processor", "machine", "logical_cpus", "physical_cpus",
                                               "total_ram_gb", "platform"))
    meta["hardware_id"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return meta


def dataset_fingerprint(database: np.ndarray, queries: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(database, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(queries, dtype=np.float32).tobytes())
    return h.hexdigest()[:16]


def generate_dataset(ds: DatasetSpec) -> Tuple[np.ndarray, np.ndarray]:
    fn = GENERATORS[ds.generator]
    return fn(num_vectors=ds.num_vectors, dimension=ds.dimension, num_queries=ds.num_queries,
              random_seed=ds.seed, **ds.params)


def profile_dataset(database: np.ndarray, ds: DatasetSpec, k: int) -> Dict[str, float]:
    """WorkloadProfiler samples with np.random; seed it (and restore the global state) for repeatability."""
    saved = np.random.get_state()
    try:
        np.random.seed(ds.seed)
        wp = WorkloadProfiler.profile(database, metric=MetricType(ds.metric), target_k=k)
    finally:
        np.random.set_state(saved)
    return {
        "profile_intrinsic_dimension": wp.intrinsic_dimension,
        "profile_hopkins": wp.clustering_tendency_hopkins,
        "profile_norm_mean": wp.norm_mean,
        "profile_norm_variance": wp.norm_variance,
        "profile_sparsity_ratio": wp.sparsity_ratio,
    }


class _Captured:
    text: str = ""


@contextlib.contextmanager
def capture_native_stderr():
    """
    Capture output written to the process-level stderr (file descriptor 2) - where FAISS's C++ code
    prints warnings such as "WARNING clustering N points to K centroids". Not thread-safe (it swaps
    fd 2 for the whole process); the collector is single-threaded. If fd-level redirection is not
    possible (e.g. no console), the block runs uncaptured and `text` stays empty.
    """
    captured = _Captured()
    try:
        sys.stderr.flush()
        saved_fd = os.dup(2)
    except (OSError, ValueError, AttributeError):
        yield captured
        return
    tmp = tempfile.TemporaryFile(mode="w+b")
    try:
        os.dup2(tmp.fileno(), 2)
        try:
            yield captured
        finally:
            try:
                sys.stderr.flush()
            except (OSError, ValueError):
                pass
            os.dup2(saved_fd, 2)
            tmp.seek(0)
            captured.text = tmp.read().decode("utf-8", "replace")
    finally:
        os.close(saved_fd)
        tmp.close()


def parse_faiss_warnings(text: str) -> List[str]:
    """Non-empty captured lines, de-duplicated in order (FAISS repeats some warnings per sub-quantizer)."""
    seen, out = set(), []
    for line in text.splitlines():
        line = line.strip()
        if line and line not in seen:
            seen.add(line)
            out.append(line)
    return out


def summarize_latency_passes(passes: List[Any]) -> Dict[str, Any]:
    """Representative (median) latency values across timing passes plus pass-to-pass variability."""
    p50 = np.array([p.latency_p50_ms for p in passes])
    p95 = np.array([p.latency_p95_ms for p in passes])
    recalls = [p.recall_at_k for p in passes]
    return {
        "latency_p50_ms": float(np.median(p50)),
        "latency_p95_ms": float(np.median(p95)),
        "latency_p99_ms": float(np.median([p.latency_p99_ms for p in passes])),
        "latency_mean_ms": float(np.median([p.latency_mean_ms for p in passes])),
        "throughput_qps_sequential": float(np.median([p.throughput_qps for p in passes])),
        "latency_repeats": len(passes),
        "latency_p50_ms_min": float(p50.min()),
        "latency_p50_ms_max": float(p50.max()),
        "latency_p95_ms_min": float(p95.min()),
        "latency_p95_ms_max": float(p95.max()),
        "latency_p95_cv": float(p95.std() / p95.mean()) if p95.mean() > 0 else 0.0,
        "recall_spread": float(max(recalls) - min(recalls)),
    }


def _params(prefix: str, cfg: Optional[IndexConfig], family: str) -> Dict[str, Any]:
    out = {f"{prefix}_{p}": "" for p in PARAM_NAMES}
    if cfg is not None:
        for p in FAMILY_PARAMS[family]:
            out[f"{prefix}_{p}"] = int(getattr(cfg, p))
    return out


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def run_collection(
    spec: CollectionSpec,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    run_id: Optional[str] = None,
    progress: Optional[Callable[[str], None]] = print,
) -> Dict[str, Any]:
    """
    Execute the benchmark grid and write `<output_dir>/<run_id>/measurements.csv` and
    `run_metadata.json`. Rows are flushed as they complete, so an interrupted run keeps its data.
    Returns the manifest dictionary.
    """
    spec.validate()
    log = progress or (lambda msg: None)
    run_id = run_id or f"{spec.name}-{_dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = os.path.join(output_dir, run_id)
    if os.path.exists(os.path.join(run_dir, MEASUREMENTS_FILE)):
        raise FileExistsError(f"run directory already contains results: {run_dir}")
    os.makedirs(run_dir, exist_ok=True)
    csv_path = os.path.join(run_dir, MEASUREMENTS_FILE)
    manifest_path = os.path.join(run_dir, MANIFEST_FILE)

    previous_threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(spec.threads)
    started = time.perf_counter()
    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "started_utc": _utc_now(),
        "spec": spec.to_dict(),
        "spec_hash": spec.spec_hash(),
        "hardware": collect_hardware_metadata(spec.threads),
        "measurement_notes": {
            "measured": ["recall_at_k", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "latency_mean_ms",
                         "throughput_qps_sequential", "build_time_s", "serialized_index_mb",
                         "latency_p50_ms_min", "latency_p50_ms_max", "latency_p95_ms_min", "latency_p95_ms_max",
                         "latency_p95_cv", "recall_spread"],
            "estimated": ["estimated_memory_mb"],
            "latency": "single-query sequential search in Python (one query per FAISS call), after warmup",
            "latency_repeats": ("each configuration is timed in `latency_repeats` back-to-back passes over all queries "
                                "on the same built index (no rebuild); latency_p50/p95/p99/mean_ms and "
                                "throughput are the MEDIAN across passes; *_min/*_max and latency_p95_cv "
                                "(std/mean of per-pass p95) describe pass-to-pass variability"),
            "recall_spread": "max - min recall across passes (0 expected: search is deterministic)",
            "faiss_warnings": ("native FAISS stderr output captured during the index build (e.g. too few "
                               "k-means training points); shared by rows with the same structure_id"),
            "throughput_qps_sequential": "queries / sum of single-query latencies; not concurrent throughput",
            "build_time_s": "per index build; rows sharing a structure_id share one build (runtime-parameter variants)",
            "serialized_index_mb": "size of faiss.write_index output, measured",
            "estimated_memory_mb": "max(serialized size, analytical model); not measured process RAM",
        },
        "datasets": [],
        "skipped_configurations": [],
        "faiss_warnings": [],
        "rows_written": 0,
        "errors": 0,
        "status": "running",
    }
    hardware_id = manifest["hardware"]["hardware_id"]
    row_id = 0

    def write_manifest():
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=2)

    write_manifest()
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for ds in spec.datasets:
                t_ds = time.perf_counter()
                metric = MetricType(ds.metric)
                database, queries = generate_dataset(ds)
                fingerprint = dataset_fingerprint(database, queries)
                profile = profile_dataset(database, ds, spec.k)
                evaluator = BenchmarkEvaluator(database, queries, metric=metric, k=spec.k)
                plan, skipped = plan_structures(spec, ds.num_vectors, ds.dimension)
                manifest["skipped_configurations"] += [f"{ds.name}: {s}" for s in skipped]
                n_rows = sum(len(s.variants) for s in plan) * spec.repeats
                log(f"[{ds.name}] {ds.num_vectors}x{ds.dimension} ({ds.generator}), fingerprint {fingerprint}: "
                    f"{len(plan)} builds, {n_rows} measurements")

                base_row = {
                    "schema_version": SCHEMA_VERSION, "run_id": run_id,
                    "dataset_name": ds.name, "dataset_generator": ds.generator,
                    "dataset_fingerprint": fingerprint, "dataset_seed": ds.seed,
                    "num_vectors": ds.num_vectors, "dimension": ds.dimension, "num_queries": ds.num_queries,
                    "metric": ds.metric, "k": spec.k, "faiss_threads": spec.threads, "hardware_id": hardware_id,
                    **profile,
                }
                for s_idx, structure in enumerate(plan):
                    fam = structure.family.value
                    structure_id = f"{ds.name}/{fam}/{s_idx}"
                    for repeat in range(spec.repeats):
                        wrapper = FaissIndexWrapper(structure.build_config, ds.dimension, metric)
                        with capture_native_stderr() as captured:
                            try:
                                build_time = wrapper.build_and_add(database)
                                build_error = None
                            except Exception as exc:  # record the failure for every variant, keep going
                                build_time = None
                                build_error = f"build failed: {type(exc).__name__}: {exc}"
                        build_warnings = parse_faiss_warnings(captured.text)
                        if build_warnings:
                            manifest["faiss_warnings"].append({"structure_id": structure_id, "repeat": repeat,
                                                               "build_config": structure.build_config.id_string,
                                                               "warnings": build_warnings})
                        build_adjustments = list(wrapper.config_adjustments) if not build_error else []

                        for variant in structure.variants:
                            row_id += 1
                            row = dict(base_row, row_id=row_id, family=fam, config_id=variant.id_string,
                                       structure_id=structure_id, repeat=repeat,
                                       **_params("req", variant, fam))
                            try:
                                if build_error:
                                    raise RuntimeError(build_error)
                                n_adj = len(wrapper.config_adjustments)
                                if fam == "HNSW":
                                    wrapper.set_runtime_parameters(efSearch=variant.efSearch)
                                elif fam in ("IVF_FLAT", "IVF_PQ"):
                                    wrapper.set_runtime_parameters(nprobe=variant.nprobe)
                                # Several timing passes on the same built index (no rebuild)
                                passes = [evaluator.measure_index(wrapper, build_time, warmup_runs=spec.warmup_runs,
                                                                  round_results=False)
                                          for _ in range(spec.latency_repeats)]
                                result = passes[0]
                                adjustments = build_adjustments + wrapper.config_adjustments[n_adj:]
                                row.update(_params("eff", wrapper.config, fam))
                                row.update({
                                    "status": "ok", "error": "",
                                    "config_adjustments": "; ".join(adjustments),
                                    "recall_at_k": float(np.median([p.recall_at_k for p in passes])),
                                    "build_time_s": float(build_time),
                                    "serialized_index_mb": float(result.index_serialized_mb),
                                    "estimated_memory_mb": result.memory_footprint_mb,
                                    "faiss_warnings": " | ".join(build_warnings),
                                })
                                row.update(summarize_latency_passes(passes))
                            except Exception as exc:
                                manifest["errors"] += 1
                                row.update(_params("eff", None, fam))
                                row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"[:500],
                                            "config_adjustments": "", "faiss_warnings": " | ".join(build_warnings)})
                                log(f"  ERROR {variant.id_string}: {row['error']}")
                            writer.writerow({c: row.get(c, "") for c in CSV_COLUMNS})
                            f.flush()
                            manifest["rows_written"] += 1
                        del wrapper
                manifest["datasets"].append({
                    "name": ds.name, "generator": ds.generator, "fingerprint": fingerprint,
                    "num_vectors": ds.num_vectors, "dimension": ds.dimension, "num_queries": ds.num_queries,
                    "seed": ds.seed, "metric": ds.metric, "params": ds.params, "profile": profile,
                    "structures_built": len(plan) * spec.repeats, "rows": n_rows,
                    "duration_s": round(time.perf_counter() - t_ds, 2),
                })
                write_manifest()
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["failure"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        raise
    finally:
        faiss.omp_set_num_threads(previous_threads)
        manifest["finished_utc"] = _utc_now()
        manifest["duration_s"] = round(time.perf_counter() - started, 2)
        manifest["files"] = {"measurements": MEASUREMENTS_FILE, "manifest": MANIFEST_FILE}
        write_manifest()
    log(f"Wrote {manifest['rows_written']} rows ({manifest['errors']} errors) to {csv_path} "
        f"in {manifest['duration_s']} s")
    manifest["run_dir"] = run_dir
    return manifest


def load_measurements(csv_path: str, include_errors: bool = False) -> List[Dict[str, Any]]:
    """Read a measurements.csv back with typed values (blank cells become None)."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            if raw["status"] != "ok" and not include_errors:
                continue
            row: Dict[str, Any] = {}
            for key, value in raw.items():
                if value == "":
                    row[key] = None
                elif key in INT_COLUMNS:
                    row[key] = int(value)
                elif key in FLOAT_COLUMNS:
                    row[key] = float(value)
                else:
                    row[key] = value
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Collect real FAISS benchmark measurements.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="default", help="built-in benchmark grid")
    parser.add_argument("--spec", help="JSON file with a CollectionSpec (overrides --preset)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="parent directory for run folders")
    parser.add_argument("--run-id", help="run folder name (default: <preset>-<UTC timestamp>)")
    parser.add_argument("--repeats", type=int, help="override independent index builds per structure")
    parser.add_argument("--latency-repeats", type=int, help="override timing passes per configuration (default 3)")
    parser.add_argument("--threads", type=int, help="override FAISS/OpenMP thread count (default from spec: 1)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan without running it")
    parser.add_argument("--write-spec", help="write the selected spec as JSON to this path and exit")
    args = parser.parse_args(argv)

    if args.spec:
        with open(args.spec, encoding="utf-8") as f:
            spec = CollectionSpec.from_dict(json.load(f))
    else:
        spec = CollectionSpec.from_dict(PRESETS[args.preset].to_dict())  # copy
    if args.repeats is not None:
        spec.repeats = args.repeats
    if args.latency_repeats is not None:
        spec.latency_repeats = args.latency_repeats
    if args.threads is not None:
        spec.threads = args.threads
    spec.validate()

    if args.write_spec:
        with open(args.write_spec, "w", encoding="utf-8") as f:
            json.dump(spec.to_dict(), f, indent=2)
        print(f"Spec written to {args.write_spec}")
        return 0

    if args.dry_run:
        print(f"Spec {spec.name!r} (hash {spec.spec_hash()}), threads={spec.threads}, repeats={spec.repeats}, "
              f"latency_repeats={spec.latency_repeats}, k={spec.k}")
        for ds in spec.datasets:
            plan, skipped = plan_structures(spec, ds.num_vectors, ds.dimension)
            per_family: Dict[str, int] = {}
            for s in plan:
                per_family[s.family.value] = per_family.get(s.family.value, 0) + len(s.variants)
            print(f"  {ds.name}: {len(plan)} builds, configs per family {per_family}, skipped {len(skipped)}")
        print(f"  total measurement rows: {count_planned_rows(spec)}")
        return 0

    run_collection(spec, output_dir=args.output_dir, run_id=args.run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
