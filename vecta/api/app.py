"""
VECTA High-Performance FastAPI REST Server.
Exposes endpoints for workload profiling, ML surrogate cost modeling,
Pareto auto-tuning, live FAISS empirical benchmarking, and vector search playground.
"""

import os
import sys
import time
import threading
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, List, Optional

# Ensure project root is available
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from vecta.config import (
    WorkloadProfile,
    IndexConfig,
    IndexFamily,
    MetricType,
    PerformancePrediction,
    BenchmarkResult
)
from vecta.profiler import WorkloadProfiler, HardwareProfiler
from vecta.cost_model import SurrogateCostModel
from vecta.tuner import AutoTuningOptimizer, RuntimeAdapter
from vecta.engine import FaissIndexWrapper, DynamicIndexManager
from vecta.benchmark import BenchmarkDatasetGenerator, BenchmarkEvaluator, BaselineComparer

app = FastAPI(
    title="Vecta ANN Auto-Tuning Kernel",
    description="Intelligent Autonomous Auto-Tuning Engine for Approximate Nearest Neighbor Search",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State Container for fast in-memory serving & playground
class AppState:
    def __init__(self):
        self.cost_model = SurrogateCostModel()
        self.optimizer = AutoTuningOptimizer(self.cost_model)
        self.current_database: Optional[np.ndarray] = None
        self.current_queries: Optional[np.ndarray] = None
        self.current_evaluator: Optional[BenchmarkEvaluator] = None
        self.dynamic_manager: Optional[DynamicIndexManager] = None
        self.current_workload: Optional[WorkloadProfile] = None
        self.active_config: Optional[IndexConfig] = None
        self.last_tuning_result: Optional[Dict[str, Any]] = None
        # Operation lock: serializes requests that build or use the dataset / active index
        # (profile, tune, benchmark, search). It can be held for seconds.
        self.lock = threading.RLock()
        # Snapshot lock: held only for the instant shared references are swapped or read, so
        # lightweight readers (e.g. /api/system) never observe a half-published state and never
        # wait behind long operations. Lock order is always lock -> snapshot_lock.
        self.snapshot_lock = threading.Lock()

state = AppState()


def _publish(**updates: Any):
    """Atomically replace shared state attributes (readers holding snapshot_lock see all or none)."""
    with state.snapshot_lock:
        for name, value in updates.items():
            setattr(state, name, value)

# Largest synthetic dataset generated per interactive request (keeps response times reasonable)
MAX_INTERACTIVE_VECTORS = 30000
MAX_K = 1024
# Upper bound on the ESTIMATED peak memory a single profile/tune/benchmark cycle may need
MAX_ESTIMATED_PEAK_MEMORY_MB = 1024.0
# Largest HNSW M in the search space (graph links dominate HNSW overhead)
_MAX_HNSW_M = 64


def estimate_peak_memory_mb(num_vectors: int, dimension: int) -> float:
    """
    Rough upper estimate (MB) of the memory one interactive cycle needs for an N x D float32
    dataset. Copies alive at the same time: the generated dataset, the exact-search (ground-truth)
    index, the deployed serving index and one benchmark index under construction, plus HNSW graph
    links for the two graph indexes. It is an estimate for admission control, not a measurement,
    and it does not include the previous dataset still held until the new one is published.
    """
    dataset_mb = num_vectors * dimension * 4.0 / (1024 * 1024)
    hnsw_links_mb = num_vectors * _MAX_HNSW_M * 2 * 4.0 / (1024 * 1024)
    return 4.0 * dataset_mb + 2.0 * hnsw_links_mb


# Request / Response Schemas
class WorkloadRequest(BaseModel):
    num_vectors: int = Field(20000, ge=1000, le=10_000_000)
    dimension: int = Field(128, ge=2, le=4096)
    metric: str = "L2"
    target_recall: float = Field(0.95, gt=0.0, le=1.0)
    max_latency_ms: float = Field(2.0, gt=0.0)
    memory_budget_mb: float = Field(128.0, gt=0.0)
    expected_qps: float = Field(500.0, ge=0.0)
    target_k: int = Field(10, ge=1, le=MAX_K)
    preset_name: Optional[str] = None

    @field_validator("metric")
    @classmethod
    def _validate_metric(cls, v: str) -> str:
        value = str(v).strip().upper()
        allowed = [m.value for m in MetricType]
        if value not in allowed:
            raise ValueError(f"metric must be one of {allowed}, got {v!r}")
        return value


class SearchRequest(BaseModel):
    query_vector: Optional[List[float]] = None
    k: int = Field(10, ge=1, le=MAX_K)
    use_random_query: bool = True


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Report malformed requests as HTTP 400 with a readable explanation."""
    errors = [
        {"loc": [str(part) for part in err.get("loc", [])], "msg": str(err.get("msg", ""))}
        for err in exc.errors()
    ]
    summary = "; ".join(f"{'.'.join(e['loc'][1:]) or 'body'}: {e['msg']}" for e in errors)
    return JSONResponse(status_code=400, content={"detail": f"Invalid request: {summary}", "errors": errors})


def _config_from_dict(cfg: Dict[str, Any]) -> IndexConfig:
    """Rebuild an IndexConfig from the optimizer's serialized recommendation."""
    return IndexConfig(
        family=IndexFamily(cfg["family"]),
        M=cfg.get("M", 16),
        efConstruction=cfg.get("efConstruction", 64),
        efSearch=cfg.get("efSearch", 32),
        nlist=cfg.get("nlist", 128),
        nprobe=cfg.get("nprobe", 8),
        m_pq=cfg.get("m_pq", 16),
        nbits_per_idx=cfg.get("nbits_per_idx", 8)
    )


def _require_consistent_active_index():
    """Raise HTTP 400 unless the active index was built for the currently profiled dataset."""
    if state.dynamic_manager is None or state.dynamic_manager.active_wrapper is None or state.active_config is None:
        raise HTTPException(
            status_code=400,
            detail="No active index for the current dataset. Run /api/tune after /api/profile."
        )
    wrapper = state.dynamic_manager.active_wrapper
    if state.current_database is None or wrapper.dimension != state.current_database.shape[1] \
            or wrapper.total_vectors != state.current_database.shape[0]:
        raise HTTPException(
            status_code=400,
            detail="The active index does not match the current dataset. Run /api/tune to rebuild it."
        )
    return wrapper


@app.on_event("startup")
async def startup_event():
    """Initialize default benchmark dataset and warm-start FAISS active index."""
    # Initialize with default clustered dataset
    n, d, q = 15000, 128, 50
    db, queries = BenchmarkDatasetGenerator.generate_gaussian_clusters(
        num_vectors=n, dimension=d, num_clusters=20, num_queries=q, random_seed=42
    )
    state.current_database = db
    state.current_queries = queries
    state.current_evaluator = BenchmarkEvaluator(db, queries, metric=MetricType.L2, k=10)
    
    # Profile default workload
    state.current_workload = WorkloadProfiler.profile(
        db, metric=MetricType.L2, target_recall=0.95, max_latency_ms=1.5, memory_budget_mb=128.0
    )
    
    # Initial Auto-tune
    tune_res = state.optimizer.tune(state.current_workload)
    winner_cfg = _config_from_dict(tune_res["recommended_config"])

    # Active Dynamic Manager (built fully, then published atomically)
    manager = DynamicIndexManager(d, MetricType.L2)
    wrapper = FaissIndexWrapper(winner_cfg, d, MetricType.L2)
    wrapper.build_and_add(db)
    manager.set_active_index(wrapper)
    _publish(last_tuning_result=tune_res, active_config=winner_cfg, dynamic_manager=manager)


@app.get("/api/system")
def get_system_telemetry():
    """Return host hardware specifications and real-time resource utilization."""
    hw = HardwareProfiler.get_hardware_profile()
    process_rss_mb = round(HardwareProfiler.current_rss_memory_mb(), 2)

    # Take one consistent reference under the snapshot lock; never re-read the shared attribute,
    # since /api/profile and /api/tune may swap it concurrently.
    with state.snapshot_lock:
        manager = state.dynamic_manager

    active_info = None
    wrapper = manager.active_wrapper if manager is not None else None
    if wrapper is not None:
        active_info = {
            "total_vectors": wrapper.total_vectors,
            "dimension": wrapper.dimension,
            "config": wrapper.config.to_dict(),
            "estimated_memory_mb": wrapper.estimated_memory_mb,
            "serialized_index_mb": round(wrapper.serialized_size_mb, 3),
            "queries_served": manager.total_queries_served
        }

    try:
        compile_options = faiss.get_compile_options().strip()
    except Exception:
        compile_options = ""

    return {
        "hardware": hw,
        "process_rss_mb": process_rss_mb,
        "engine": {
            "name": "FAISS",
            "version": getattr(faiss, "__version__", "unknown"),
            "compile_options": compile_options
        },
        "active_index": active_info
    }


@app.get("/api/presets")
def get_presets():
    """Return list of research & industry scenario presets for panel demo."""
    presets = [
        WorkloadProfiler.get_preset("rag_semantic_search"),
        WorkloadProfiler.get_preset("recommendation_high_qps"),
        WorkloadProfiler.get_preset("edge_memory_constrained"),
        WorkloadProfiler.get_preset("sift_benchmark")
    ]
    return {"presets": presets}


@app.post("/api/profile")
def profile_workload(req: WorkloadRequest):
    """Profile vector manifold geometry and register SLA targets."""
    with state.lock:
        warnings: List[str] = []
        # Synthesize dataset corresponding to requested scale (capped to keep responses interactive)
        n = min(req.num_vectors, MAX_INTERACTIVE_VECTORS)
        if n < req.num_vectors:
            warnings.append(
                f"num_vectors={req.num_vectors} exceeds the interactive limit; "
                f"profiled and tuned a synthetic dataset of {n} vectors instead."
            )
        d = req.dimension
        if req.target_k > n:
            raise HTTPException(status_code=400, detail=f"target_k={req.target_k} exceeds the dataset size ({n}).")

        # Admission control on the estimated total memory of the whole cycle, not per-field bounds
        peak_mb = estimate_peak_memory_mb(n, d)
        if peak_mb > MAX_ESTIMATED_PEAK_MEMORY_MB:
            raise HTTPException(
                status_code=400,
                detail=(f"Estimated peak memory for N={n}, D={d} is {peak_mb:.0f} MB (dataset, ground-truth "
                        f"index, serving index and a benchmark index), above the "
                        f"{MAX_ESTIMATED_PEAK_MEMORY_MB:.0f} MB interactive limit. Reduce num_vectors or dimension.")
            )

        # Build everything locally first; shared state is only replaced once all steps succeed,
        # so a failure here leaves the previous dataset and serving index intact.
        try:
            db, queries = BenchmarkDatasetGenerator.generate_gaussian_clusters(
                num_vectors=n, dimension=d, num_clusters=max(4, n // 500), num_queries=50, random_seed=42
            )
            metric = MetricType(req.metric)
            evaluator = BenchmarkEvaluator(db, queries, metric=metric, k=req.target_k)
            wp = WorkloadProfiler.profile(
                db,
                metric=metric,
                target_recall=req.target_recall,
                max_latency_ms=req.max_latency_ms,
                memory_budget_mb=req.memory_budget_mb,
                expected_qps=req.expected_qps,
                target_k=req.target_k
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Profiling failed ({type(exc).__name__}: {exc}); the previous dataset and index were kept."
            )

        # Publish the new dataset; the previous serving index / recommendation belong to the old
        # dataset, so they are invalidated in the same atomic step.
        _publish(
            current_database=db,
            current_queries=queries,
            current_evaluator=evaluator,
            current_workload=wp,
            dynamic_manager=None,
            active_config=None,
            last_tuning_result=None
        )

        return {
            "workload_profile": wp.to_dict(),
            "warnings": warnings,
            "estimated_peak_memory_mb": round(peak_mb, 1),
            "vector_sample_2d": db[:300, :2].tolist()  # Sample 2D slice for canvas rendering
        }


@app.post("/api/tune")
def run_auto_tuning(req: Optional[WorkloadRequest] = None):
    """Run surrogate model prediction, compute Pareto frontier, and select optimal configuration."""
    with state.lock:
        if req is not None:
            profile_workload(req)

        wp = state.current_workload
        if wp is None or state.current_database is None:
            raise HTTPException(status_code=400, detail="No active workload profiled. Call /api/profile first.")

        # Tune and rebuild locally; nothing shared changes unless every step succeeds, so a failed
        # rebuild can never leave active_config describing a different index than the one serving.
        try:
            tune_res = state.optimizer.tune(wp)
            winner_cfg = _config_from_dict(tune_res["recommended_config"])

            wrapper = FaissIndexWrapper(winner_cfg, wp.dimension, wp.metric)
            wrapper.build_and_add(state.current_database)
            manager = DynamicIndexManager(wp.dimension, wp.metric)
            manager.set_active_index(wrapper)
        except Exception as exc:
            serving = state.dynamic_manager.active_wrapper.config.id_string \
                if state.dynamic_manager is not None and state.dynamic_manager.active_wrapper is not None else None
            kept = (f"the previously deployed index {serving} is still serving" if serving
                    else "no index is currently active for this dataset")
            raise HTTPException(
                status_code=500,
                detail=f"Index rebuild failed ({type(exc).__name__}: {exc}); state unchanged: {kept}."
            )

        # Report exactly what FAISS built (identical to the recommendation unless adjustments were needed)
        tune_res["deployed_config"] = wrapper.config.to_dict()
        tune_res["config_adjustments"] = list(wrapper.config_adjustments)

        # Recommendation, config and serving index are published together, never separately
        _publish(active_config=winner_cfg, dynamic_manager=manager, last_tuning_result=tune_res)
        return tune_res


@app.post("/api/benchmark")
def run_arena_benchmark():
    """Execute live empirical comparison: FAISS Defaults vs Vecta Auto-Tuned."""
    with state.lock:
        if state.current_evaluator is None:
            raise HTTPException(status_code=400, detail="No profiled dataset to benchmark. Call /api/profile first.")
        _require_consistent_active_index()
        if state.current_evaluator.dimension != state.current_database.shape[1]:
            raise HTTPException(status_code=400, detail="Benchmark evaluator does not match the current dataset.")

        report = BaselineComparer.run_head_to_head_arena(
            state.current_evaluator, state.active_config, workload=state.current_workload
        )
        return report


@app.post("/api/search")
def search_vector(req: SearchRequest):
    """Query active index in real time and return nearest neighbors with distance & GT accuracy."""
    with state.lock:
        wrapper = _require_consistent_active_index()
        dim = wrapper.dimension

        if req.k > wrapper.total_vectors:
            raise HTTPException(status_code=400, detail=f"k={req.k} exceeds the number of indexed vectors ({wrapper.total_vectors}).")

        explicit_random = "use_random_query" in req.model_fields_set and req.use_random_query
        if req.query_vector is not None and explicit_random:
            raise HTTPException(status_code=400, detail="Provide either query_vector or use_random_query=true, not both.")
        if req.query_vector is None and not req.use_random_query:
            raise HTTPException(status_code=400, detail="use_random_query=false requires a query_vector.")

        if req.query_vector is not None:
            if len(req.query_vector) != dim:
                raise HTTPException(
                    status_code=400,
                    detail=f"query_vector has {len(req.query_vector)} dimensions; the active index expects {dim}."
                )
            q_vec = np.array([req.query_vector], dtype=np.float32)
            if not np.all(np.isfinite(q_vec)):
                raise HTTPException(status_code=400, detail="query_vector must contain only finite numbers.")
            query_source = "custom"
        elif state.current_queries is not None and len(state.current_queries) > 0:
            # Sample random query from the held-out query pool
            rand_idx = np.random.randint(0, len(state.current_queries))
            q_vec = state.current_queries[rand_idx:rand_idx + 1]
            query_source = "random_pool"
        else:
            q_vec = np.random.randn(1, dim).astype(np.float32)
            q_vec /= np.linalg.norm(q_vec)
            query_source = "random_generated"

        res = state.dynamic_manager.query(q_vec, k=req.k)

        # Ground-truth accuracy: exact brute-force search for *this* query over the indexed dataset
        exact_match_count = None
        accuracy_pct = None
        evaluator = state.current_evaluator
        if evaluator is not None and evaluator.dimension == dim:
            _, gt_idx = evaluator.exact_search(q_vec, k=req.k)
            true_top_k = {int(i) for i in gt_idx[0] if i >= 0}
            retrieved = {int(i) for i in res["indices"][0] if i >= 0}
            exact_match_count = len(retrieved.intersection(true_top_k))
            accuracy_pct = round((exact_match_count / req.k) * 100.0, 1)

        res["k"] = req.k
        res["metric"] = wrapper.metric.value
        res["query_source"] = query_source
        res["query_vector_sample"] = q_vec[0, :8].tolist()
        res["exact_matches_in_top_k"] = exact_match_count
        res["accuracy_pct"] = accuracy_pct
        res["ground_truth_available"] = exact_match_count is not None

        return res


# Mount static directory for modern Web UI
static_path = os.path.join(os.path.dirname(__file__), "..", "ui", "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(static_path, "index.html"))
