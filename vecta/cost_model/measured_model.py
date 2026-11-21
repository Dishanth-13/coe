"""
VECTA Measured Cost Model.

Random Forest cost model trained on REAL FAISS measurements collected by
`vecta.benchmark.collector` (instead of the synthetic formulas used by SurrogateCostModel,
which is left unchanged for comparison).

Design (kept deliberately simple and explainable):
  * One model per index family and target, because each family has different cost drivers.
  * Targets: measured Recall@k, log(p95 latency), log(p50 latency), log(build time).
    Latency/build are modelled in log space (multiplicative errors, wide ranges).
  * Build time is trained on unique builds (rows sharing a structure_id share one build).
  * Flat: recall is exactly 1 by definition; latency/build use a log-log fit on N*D because only
    one Flat row exists per dataset.
  * Memory is not learned: `memory_estimator` computes it from the index structure.
  * Configurations FAISS reports as under-trained (too few k-means points per centroid) carry an
    explicit feature and are flagged in predictions.
"""

import glob
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from vecta.config import IndexConfig, IndexFamily
from vecta.cost_model.memory_estimator import estimate_index_memory_mb

# FAISS k-means warns below this many training points per centroid (Clustering min_points_per_centroid)
FAISS_MIN_POINTS_PER_CENTROID = 39

LEARNED_FAMILIES = (IndexFamily.HNSW, IndexFamily.IVF_FLAT, IndexFamily.IVF_PQ)
TARGETS = ("recall", "log_p95", "log_p50")

RF_PARAMS = dict(n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=1)


@dataclass
class DatasetFeatures:
    """Dataset-level inputs available before benchmarking (from the workload profile)."""
    num_vectors: int
    dimension: int
    intrinsic_dimension: float
    k: int = 10


@dataclass
class MeasuredPrediction:
    family: str
    config_id: str
    predicted_recall: float
    predicted_latency_p50_ms: float
    predicted_latency_p95_ms: float
    predicted_build_time_s: float
    estimated_memory_mb: float          # index-specific calculation, not ML
    undertrained: bool                  # FAISS would warn about too few training points


def config_from_row(row: Dict[str, Any], prefix: str = "eff") -> IndexConfig:
    family = IndexFamily(row["family"])
    params = {}
    for name in ("M", "efConstruction", "efSearch", "nlist", "nprobe", "m_pq", "nbits_per_idx"):
        value = row.get(f"{prefix}_{name}")
        if value is not None:
            params[name] = int(value)
    return IndexConfig(family=family, **params)


def dataset_from_row(row: Dict[str, Any]) -> DatasetFeatures:
    return DatasetFeatures(int(row["num_vectors"]), int(row["dimension"]),
                           float(row["profile_intrinsic_dimension"]), int(row["k"]))


def is_undertrained(ds: DatasetFeatures, cfg: IndexConfig) -> bool:
    """True when FAISS k-means would have fewer than 39 training points per centroid."""
    if cfg.family in (IndexFamily.IVF_FLAT, IndexFamily.IVF_PQ):
        if ds.num_vectors < FAISS_MIN_POINTS_PER_CENTROID * cfg.nlist:
            return True
    if cfg.family == IndexFamily.IVF_PQ:
        if ds.num_vectors < FAISS_MIN_POINTS_PER_CENTROID * (1 << cfg.nbits_per_idx):
            return True
    return False


def _common(ds: DatasetFeatures) -> List[float]:
    return [np.log2(ds.num_vectors), float(ds.dimension), np.log2(ds.dimension),
            float(ds.intrinsic_dimension), ds.intrinsic_dimension / ds.dimension, np.log2(ds.k)]


COMMON_NAMES = ["log2_n", "dimension", "log2_dimension", "intrinsic_dimension", "intrinsic_ratio", "log2_k"]

FEATURE_NAMES = {
    IndexFamily.HNSW: COMMON_NAMES + ["M", "log2_efConstruction", "log2_efSearch", "log2_efSearch_over_k",
                                      "log2_search_cost"],
    IndexFamily.IVF_FLAT: COMMON_NAMES + ["log2_nlist", "log2_nprobe", "probe_fraction", "log2_points_per_list",
                                          "log2_scan_cost", "undertrained"],
    IndexFamily.IVF_PQ: COMMON_NAMES + ["log2_nlist", "log2_nprobe", "probe_fraction", "log2_points_per_list",
                                        "m_pq", "bits_per_dimension", "log2_scan_cost", "undertrained"],
}

BUILD_FEATURE_NAMES = {
    IndexFamily.HNSW: ["log2_n", "dimension", "M", "log2_efConstruction", "log2_build_cost"],
    IndexFamily.IVF_FLAT: ["log2_n", "dimension", "log2_nlist", "log2_build_cost"],
    IndexFamily.IVF_PQ: ["log2_n", "dimension", "log2_nlist", "m_pq", "log2_build_cost"],
}


def features(ds: DatasetFeatures, cfg: IndexConfig) -> List[float]:
    """Search-time features. Cost terms approximate the number of float operations per query."""
    n, d = ds.num_vectors, ds.dimension
    if cfg.family == IndexFamily.HNSW:
        return _common(ds) + [float(cfg.M), np.log2(cfg.efConstruction), np.log2(cfg.efSearch),
                              np.log2(cfg.efSearch / ds.k), np.log2(cfg.efSearch * cfg.M * d)]
    if cfg.family in (IndexFamily.IVF_FLAT, IndexFamily.IVF_PQ):
        per_list = n / cfg.nlist
        scanned = cfg.nprobe * per_list
        ivf = [np.log2(cfg.nlist), np.log2(cfg.nprobe), cfg.nprobe / cfg.nlist, np.log2(per_list)]
        undertrained = float(is_undertrained(ds, cfg))
        if cfg.family == IndexFamily.IVF_FLAT:
            return _common(ds) + ivf + [np.log2(scanned * d + cfg.nlist * d), undertrained]
        bits_per_dim = cfg.m_pq * cfg.nbits_per_idx / d
        scan_cost = scanned * cfg.m_pq + cfg.nlist * d + (1 << cfg.nbits_per_idx) * d  # ADC tables + scan
        return _common(ds) + ivf + [float(cfg.m_pq), bits_per_dim, np.log2(scan_cost), undertrained]
    raise ValueError(f"no learned features for {cfg.family}")


def build_features(ds: DatasetFeatures, cfg: IndexConfig) -> List[float]:
    """Build-time features use structural parameters only (runtime efSearch/nprobe do not affect builds)."""
    n, d = ds.num_vectors, ds.dimension
    if cfg.family == IndexFamily.HNSW:
        return [np.log2(n), float(d), float(cfg.M), np.log2(cfg.efConstruction),
                np.log2(n * np.log2(n) * cfg.efConstruction * cfg.M * d)]
    if cfg.family == IndexFamily.IVF_FLAT:
        return [np.log2(n), float(d), np.log2(cfg.nlist), np.log2(n * cfg.nlist * d)]
    if cfg.family == IndexFamily.IVF_PQ:
        return [np.log2(n), float(d), np.log2(cfg.nlist), float(cfg.m_pq),
                np.log2(n * cfg.nlist * d + n * (1 << cfg.nbits_per_idx) * d)]
    raise ValueError(f"no build features for {cfg.family}")


class _LogLinear:
    """log(y) = a + b * log(x): used for Flat, where only one measurement per dataset exists."""

    def fit(self, x: Sequence[float], y: Sequence[float]) -> "_LogLinear":
        lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
        if len(set(lx)) >= 2:
            self.b, self.a = np.polyfit(lx, ly, 1)
        else:  # single point: assume linear scaling (b = 1)
            self.b, self.a = 1.0, float(np.mean(ly - lx))
        return self

    def predict(self, x: float) -> float:
        return float(np.exp(self.a + self.b * np.log(x)))


class MeasuredCostModel:
    """Per-family Random Forests trained on measured benchmark rows (collector CSV, status == ok)."""

    def __init__(self):
        self.models: Dict[IndexFamily, Dict[str, RandomForestRegressor]] = {}
        self.build_models: Dict[IndexFamily, RandomForestRegressor] = {}
        self.flat_p95 = self.flat_p50 = self.flat_build = None
        self.training_summary: Dict[str, Any] = {}

    def fit(self, rows: List[Dict[str, Any]]) -> "MeasuredCostModel":
        rows = [r for r in rows if r.get("status", "ok") == "ok"]
        if not rows:
            raise ValueError("no measurement rows to train on")
        summary = {"rows": len(rows), "datasets": sorted({r["dataset_name"] for r in rows}), "families": {}}

        for fam in LEARNED_FAMILIES:
            fam_rows = [r for r in rows if r["family"] == fam.value]
            if len(fam_rows) < 4:
                continue
            X = np.array([features(dataset_from_row(r), config_from_row(r)) for r in fam_rows])
            y = {
                "recall": np.array([r["recall_at_k"] for r in fam_rows]),
                "log_p95": np.log(np.array([r["latency_p95_ms"] for r in fam_rows])),
                "log_p50": np.log(np.array([r["latency_p50_ms"] for r in fam_rows])),
            }
            self.models[fam] = {t: RandomForestRegressor(**RF_PARAMS).fit(X, y[t]) for t in TARGETS}

            builds = {}
            for r in fam_rows:  # one sample per build
                builds.setdefault((r["structure_id"], r["repeat"]), r)
            Xb = np.array([build_features(dataset_from_row(r), config_from_row(r)) for r in builds.values()])
            yb = np.log(np.array([r["build_time_s"] for r in builds.values()]))
            self.build_models[fam] = RandomForestRegressor(**RF_PARAMS).fit(Xb, yb)
            summary["families"][fam.value] = {"rows": len(fam_rows), "builds": len(builds),
                                              "undertrained_rows": int(X[:, FEATURE_NAMES[fam].index("undertrained")].sum())
                                              if "undertrained" in FEATURE_NAMES[fam] else 0}

        flat = [r for r in rows if r["family"] == IndexFamily.FLAT.value]
        if flat:
            size = [r["num_vectors"] * r["dimension"] for r in flat]
            self.flat_p95 = _LogLinear().fit(size, [r["latency_p95_ms"] for r in flat])
            self.flat_p50 = _LogLinear().fit(size, [r["latency_p50_ms"] for r in flat])
            self.flat_build = _LogLinear().fit(size, [r["build_time_s"] for r in flat])
            summary["families"]["FLAT"] = {"rows": len(flat), "builds": len(flat), "undertrained_rows": 0}
        self.training_summary = summary
        return self

    def supports(self, family: IndexFamily) -> bool:
        return family in self.models or (family == IndexFamily.FLAT and self.flat_p95 is not None)

    def predict(self, ds: DatasetFeatures, cfg: IndexConfig) -> MeasuredPrediction:
        if not self.supports(cfg.family):
            raise ValueError(f"model was not trained for {cfg.family.value}")
        memory = estimate_index_memory_mb(cfg.family, ds.num_vectors, ds.dimension, cfg)
        if cfg.family == IndexFamily.FLAT:
            size = ds.num_vectors * ds.dimension
            return MeasuredPrediction(cfg.family.value, cfg.id_string, 1.0, self.flat_p50.predict(size),
                                      self.flat_p95.predict(size), self.flat_build.predict(size), memory, False)
        x = np.array([features(ds, cfg)])
        m = self.models[cfg.family]
        recall = float(np.clip(m["recall"].predict(x)[0], 0.0, 1.0))
        build = float(np.exp(self.build_models[cfg.family].predict(np.array([build_features(ds, cfg)]))[0]))
        return MeasuredPrediction(
            family=cfg.family.value, config_id=cfg.id_string, predicted_recall=recall,
            predicted_latency_p50_ms=float(np.exp(m["log_p50"].predict(x)[0])),
            predicted_latency_p95_ms=float(np.exp(m["log_p95"].predict(x)[0])),
            predicted_build_time_s=build, estimated_memory_mb=memory,
            undertrained=is_undertrained(ds, cfg))

    def predict_batch(self, ds: DatasetFeatures, configs: Sequence[IndexConfig]) -> List[Optional[MeasuredPrediction]]:
        """
        Predict many configurations with one forest call per family (fast for a full candidate grid).
        Entries are None for families the model was not trained on (callers fall back for those).
        """
        out: List[Optional[MeasuredPrediction]] = [None] * len(configs)
        by_family: Dict[IndexFamily, List[int]] = {}
        for i, cfg in enumerate(configs):
            by_family.setdefault(cfg.family, []).append(i)
        for fam, idx in by_family.items():
            if not self.supports(fam):
                continue
            if fam == IndexFamily.FLAT:
                for i in idx:
                    out[i] = self.predict(ds, configs[i])
                continue
            X = np.array([features(ds, configs[i]) for i in idx])
            Xb = np.array([build_features(ds, configs[i]) for i in idx])
            m = self.models[fam]
            recall = np.clip(m["recall"].predict(X), 0.0, 1.0)
            p95 = np.exp(m["log_p95"].predict(X))
            p50 = np.exp(m["log_p50"].predict(X))
            build = np.exp(self.build_models[fam].predict(Xb))
            for j, i in enumerate(idx):
                cfg = configs[i]
                out[i] = MeasuredPrediction(
                    family=fam.value, config_id=cfg.id_string, predicted_recall=float(recall[j]),
                    predicted_latency_p50_ms=float(p50[j]), predicted_latency_p95_ms=float(p95[j]),
                    predicted_build_time_s=float(build[j]),
                    estimated_memory_mb=estimate_index_memory_mb(fam, ds.num_vectors, ds.dimension, cfg),
                    undertrained=is_undertrained(ds, cfg))
        return out

    @classmethod
    def from_csv(cls, csv_path: str) -> "MeasuredCostModel":
        from vecta.benchmark.collector import load_measurements
        model = cls().fit(load_measurements(csv_path))
        model.training_summary["source"] = os.path.abspath(csv_path)
        return model

    def predict_row(self, row: Dict[str, Any]) -> MeasuredPrediction:
        return self.predict(dataset_from_row(row), config_from_row(row))

    def feature_importances(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        out = {}
        for fam, models in self.models.items():
            out[fam.value] = {t: dict(sorted(zip(FEATURE_NAMES[fam], m.feature_importances_.round(3)),
                                             key=lambda kv: -kv[1])) for t, m in models.items()}
        return out


# ---------------------------------------------------------------------------
# Locating training data and loading (cached) for the API / optimizer
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ENV_MEASUREMENTS = "VECTA_MEASUREMENTS_CSV"     # explicit training data
ENV_DISABLE = "VECTA_DISABLE_MEASURED_MODEL"    # "1" forces the synthetic fallback
_MODEL_CACHE: Dict[Tuple[str, float], MeasuredCostModel] = {}
_CACHE_LOCK = threading.Lock()


def default_measurements_path() -> Optional[str]:
    """
    Training data search order: $VECTA_MEASUREMENTS_CSV, then a committed reference run under
    benchmarks/reference/, then the newest completed schema-v2 run under benchmarks/results/.
    Returns None when nothing is available.
    """
    explicit = os.environ.get(ENV_MEASUREMENTS)
    if explicit:
        return explicit
    reference = sorted(glob.glob(os.path.join(PROJECT_ROOT, "benchmarks", "reference", "*", "measurements.csv")))
    if reference:
        return reference[-1]
    from vecta.cost_model.evaluation import find_latest_measurements
    try:
        return find_latest_measurements(os.path.join(PROJECT_ROOT, "benchmarks", "results"))
    except FileNotFoundError:
        return None


def load_measured_model(csv_path: Optional[str] = None) -> Tuple[Optional[MeasuredCostModel], Optional[str]]:
    """
    Return (model, None) or (None, reason). Training takes a few seconds, so models are cached per
    (file, modification time) for the life of the process.
    """
    if os.environ.get(ENV_DISABLE) == "1":
        return None, f"measured model disabled by {ENV_DISABLE}=1"
    path = csv_path or default_measurements_path()
    if not path:
        return None, "no benchmark measurements found (run: python -m vecta.benchmark.collector)"
    if not os.path.exists(path):
        return None, f"benchmark measurements not found: {path}"
    key = (os.path.abspath(path), os.path.getmtime(path))
    with _CACHE_LOCK:
        if key not in _MODEL_CACHE:
            try:
                _MODEL_CACHE[key] = MeasuredCostModel.from_csv(path)
            except Exception as exc:
                return None, f"could not train measured model from {path}: {type(exc).__name__}: {exc}"
        return _MODEL_CACHE[key], None
