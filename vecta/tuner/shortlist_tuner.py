"""
VECTA Shortlist-and-Benchmark Tuner.

  1. Predict every candidate configuration with the measured-data cost model
     (MeasuredCostModel; falls back to the synthetic SurrogateCostModel per candidate when the
     measured model is unavailable or does not support a family).
  2. Shortlist the most promising candidates (memory budget enforced with the index-specific
     memory calculation; FAISS-under-trained configurations excluded).
  3. Benchmark the shortlist with real FAISS on the actual dataset (BenchmarkEvaluator).
  4. Select the recommendation from MEASURED results under the recall and memory constraints.

The original prediction-only AutoTuningOptimizer is kept unchanged and is used when no dataset
is available for benchmarking, or when every shortlisted benchmark fails.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from vecta.config import IndexConfig, IndexFamily, PerformancePrediction, WorkloadProfile
from vecta.benchmark.evaluator import BenchmarkEvaluator
from vecta.benchmark.collector import summarize_latency_passes
from vecta.cost_model.measured_model import (DatasetFeatures, MeasuredCostModel, is_undertrained,
                                             load_measured_model)
from vecta.cost_model.memory_estimator import estimate_index_memory_mb
from vecta.engine.faiss_wrapper import FaissIndexWrapper
from vecta.tuner.optimizer import AutoTuningOptimizer, format_against_target, _fmt_percent, _fmt_ms, _fmt_mb

DEFAULT_SHORTLIST_SIZE = 8
DEFAULT_MAX_PER_FAMILY = 3
DEFAULT_LATENCY_PASSES = 3   # timing passes per benchmarked config; the median is reported
# Measured p95 latencies within this fraction of the fastest feasible one are treated as tied
# (single-query timing varies a few percent run to run); ties go to the smallest index.
DEFAULT_TIE_TOLERANCE = 0.05
SOURCE_MEASURED = "measured_model"
SOURCE_SYNTHETIC = "synthetic_model"


def structure_key(cfg: IndexConfig) -> Tuple:
    """Configurations with the same key share one index build (they differ only in efSearch/nprobe)."""
    if cfg.family == IndexFamily.HNSW:
        return (cfg.family.value, cfg.M, cfg.efConstruction)
    if cfg.family == IndexFamily.IVF_FLAT:
        return (cfg.family.value, cfg.nlist)
    if cfg.family == IndexFamily.IVF_PQ:
        return (cfg.family.value, cfg.nlist, cfg.m_pq, cfg.nbits_per_idx)
    return (cfg.family.value,)


def shortlist_candidates(predictions: Sequence[Dict[str, Any]], workload: WorkloadProfile,
                         size: int, families: Optional[Sequence[str]] = None,
                         max_per_family: Optional[int] = DEFAULT_MAX_PER_FAMILY) -> Tuple[List[int], Dict[int, str]]:
    """
    Choose up to `size` candidate indices to benchmark.
      * excluded: estimated index memory above the budget, or FAISS under-trained (k-means warning);
      * order: candidates predicted to reach the recall target, fastest predicted p95 first; then the
        remaining candidates by highest predicted recall (closest to the target);
      * diversity: a first pass takes at most `max_per_family` per index family (needs `families`),
        so near-duplicates of one family cannot fill the whole shortlist; free slots are then filled
        in the same order.
    If every candidate exceeds the memory budget, the smallest ones are shortlisted (best effort).
    Returns (shortlist, {index: exclusion reason}).
    """
    excluded: Dict[int, str] = {}
    eligible = []
    for i, p in enumerate(predictions):
        if p["undertrained"]:
            excluded[i] = "under-trained: too few k-means training points per centroid"
        elif p["memory_mb"] > workload.memory_budget_mb:
            excluded[i] = "estimated index memory exceeds budget"
        else:
            eligible.append(i)
    if not eligible:
        # The memory budget is a hard constraint; under-training is a quality warning whose effect the
        # benchmark measures. So: within budget even if under-trained, else the smallest indexes.
        within_budget = [i for i, p in enumerate(predictions) if p["memory_mb"] <= workload.memory_budget_mb]
        if within_budget:
            for i in within_budget:
                excluded.pop(i, None)
            eligible = within_budget
        else:
            return sorted(range(len(predictions)), key=lambda i: predictions[i]["memory_mb"])[:size], excluded
    predicted_ok = sorted((i for i in eligible if predictions[i]["recall"] >= workload.target_recall),
                          key=lambda i: predictions[i]["latency_p95_ms"])
    rest = sorted((i for i in eligible if predictions[i]["recall"] < workload.target_recall),
                  key=lambda i: -predictions[i]["recall"])
    ordered = predicted_ok + rest
    if families is None or max_per_family is None:
        return ordered[:size], excluded
    chosen: List[int] = []
    per_family: Dict[str, int] = {}
    for i in ordered:  # diversity pass
        if len(chosen) == size:
            break
        if per_family.get(families[i], 0) < max_per_family:
            chosen.append(i)
            per_family[families[i]] = per_family.get(families[i], 0) + 1
    for i in ordered:  # fill remaining slots in priority order
        if len(chosen) == size:
            break
        if i not in chosen:
            chosen.append(i)
    return sorted(chosen, key=ordered.index), excluded


def select_measured_winner(measured: Dict[Any, Dict[str, Any]], checks: Dict[Any, Dict[str, bool]],
                           tie_tolerance: float = DEFAULT_TIE_TOLERANCE) -> Tuple[Any, Dict[str, Any]]:
    """
    Choose the recommendation from MEASURED results.
      * Feasible = measured recall and serialized index size meet the targets.
      * Among feasible candidates, every one whose measured p95 is within `tie_tolerance` of the
        fastest is considered tied (differences inside timing noise). If the fastest meets the latency
        limit, tied candidates must meet it too. Ties are broken deterministically: smallest measured
        index, then configuration id - so repeated runs on the same workload agree.
      * No feasible candidate: best effort = highest recall within the memory budget, else smallest index.
    Returns (winner key, tie-break details).
    """
    feasible = [i for i in measured if checks[i]["recall"] and checks[i]["memory"]]
    if not feasible:
        in_memory = [i for i in measured if checks[i]["memory"]]
        if in_memory:
            winner = max(in_memory, key=lambda i: (measured[i]["recall_at_k"], -measured[i]["latency_p95_ms"]))
        else:
            winner = min(measured, key=lambda i: measured[i]["serialized_index_mb"])
        return winner, {"applied": False, "reason": "no feasible configuration"}

    fastest = min(feasible, key=lambda i: (measured[i]["latency_p95_ms"], measured[i]["effective_config"]))
    limit = measured[fastest]["latency_p95_ms"] * (1.0 + max(0.0, tie_tolerance))
    tied = [i for i in feasible if measured[i]["latency_p95_ms"] <= limit
            and (checks[i]["latency"] or not checks[fastest]["latency"])]
    winner = min(tied, key=lambda i: (measured[i]["serialized_index_mb"], measured[i]["effective_config"]))
    details = {
        "applied": winner != fastest,
        "tolerance": tie_tolerance,
        "fastest_config": measured[fastest]["effective_config"],
        "fastest_p95_ms": measured[fastest]["latency_p95_ms"],
        "tied_configs": sorted(measured[i]["effective_config"] for i in tied),
        "rule": "within tolerance of the fastest measured p95, choose the smallest serialized index",
    }
    return winner, details


class ShortlistTuner:
    """Predict -> shortlist -> benchmark -> select on measurements."""

    def __init__(
        self,
        synthetic_optimizer: Optional[AutoTuningOptimizer] = None,
        measured_model: Optional[MeasuredCostModel] = None,
        model_loader: Optional[Callable[[], Tuple[Optional[MeasuredCostModel], Optional[str]]]] = load_measured_model,
        shortlist_size: int = DEFAULT_SHORTLIST_SIZE,
        max_per_family: Optional[int] = DEFAULT_MAX_PER_FAMILY,
        latency_passes: int = DEFAULT_LATENCY_PASSES,
        tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
    ):
        if not 1 <= shortlist_size <= 10:
            raise ValueError("shortlist_size must be between 1 and 10")
        self.synthetic = synthetic_optimizer or AutoTuningOptimizer()
        self.measured_model = measured_model
        self.model_loader = model_loader
        self.shortlist_size = shortlist_size
        self.max_per_family = max_per_family
        self.latency_passes = max(1, int(latency_passes))
        if tie_tolerance < 0:
            raise ValueError("tie_tolerance must be >= 0")
        self.tie_tolerance = float(tie_tolerance)

    # ------------------------------------------------------------------ predictions
    def _resolve_model(self) -> Tuple[Optional[MeasuredCostModel], Optional[str]]:
        if self.measured_model is not None:
            return self.measured_model, None
        if self.model_loader is None:
            return None, "no measured model configured"
        try:
            return self.model_loader()
        except Exception as exc:
            return None, f"measured model loading failed: {type(exc).__name__}: {exc}"

    def predict_candidates(self, workload: WorkloadProfile, candidates: Sequence[IndexConfig]
                           ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Per-candidate predictions with their source; memory is always the index-specific calculation."""
        ds = DatasetFeatures(workload.num_vectors, workload.dimension, workload.intrinsic_dimension,
                             workload.target_k)
        model, reason = self._resolve_model()
        measured: List[Optional[Any]] = [None] * len(candidates)
        if model is not None:
            try:
                measured = model.predict_batch(ds, candidates)
            except Exception as exc:
                reason = f"measured model prediction failed: {type(exc).__name__}: {exc}"
                measured = [None] * len(candidates)

        preds: List[Dict[str, Any]] = []
        for cfg, mp in zip(candidates, measured):
            memory = estimate_index_memory_mb(cfg.family, workload.num_vectors, workload.dimension, cfg)
            undertrained = is_undertrained(ds, cfg)
            if mp is not None:
                preds.append({"source": SOURCE_MEASURED, "recall": mp.predicted_recall,
                              "latency_p50_ms": mp.predicted_latency_p50_ms,
                              "latency_p95_ms": mp.predicted_latency_p95_ms,
                              "build_time_s": mp.predicted_build_time_s,
                              "memory_mb": memory, "undertrained": undertrained})
            else:
                sp = self.synthetic.cost_model.predict(workload, cfg)
                preds.append({"source": SOURCE_SYNTHETIC, "recall": sp.predicted_recall,
                              "latency_p50_ms": sp.predicted_latency_p50,
                              "latency_p95_ms": sp.predicted_latency_p95,
                              "build_time_s": sp.predicted_build_time_s,
                              "memory_mb": memory, "undertrained": undertrained})
        if model is not None and reason is None and any(p["source"] == SOURCE_SYNTHETIC for p in preds):
            unsupported = sorted({c.family.value for c, p in zip(candidates, preds) if p["source"] == SOURCE_SYNTHETIC})
            reason = f"measured model does not support {', '.join(unsupported)}; synthetic predictions used for those"
        return preds, reason

    @staticmethod
    def _as_performance_prediction(p: Dict[str, Any], workload: WorkloadProfile) -> PerformancePrediction:
        meets = (p["recall"] >= workload.target_recall, p["latency_p95_ms"] <= workload.max_latency_ms,
                 p["memory_mb"] <= workload.memory_budget_mb)
        return PerformancePrediction(
            predicted_recall=float(p["recall"]), predicted_latency_p50=float(p["latency_p50_ms"]),
            predicted_latency_p95=float(p["latency_p95_ms"]), predicted_latency_p99=float(p["latency_p95_ms"]),
            predicted_memory_mb=float(p["memory_mb"]), predicted_build_time_s=float(p["build_time_s"]),
            recall_uncertainty=None, latency_uncertainty_p95=None, memory_uncertainty=None,
            is_feasible=all(meets), meets_recall_target=meets[0], meets_latency_target=meets[1],
            meets_memory_budget=meets[2])

    # ------------------------------------------------------------------ benchmarking
    @staticmethod
    def benchmark(evaluator: BenchmarkEvaluator, configs: Sequence[IndexConfig],
                  latency_passes: int = DEFAULT_LATENCY_PASSES) -> List[Dict[str, Any]]:
        """
        Measure configs on the evaluator's dataset; one build per structure_key, `latency_passes`
        timing passes per config (median reported, as in the benchmark collector). Errors are recorded.
        """
        results: List[Dict[str, Any]] = [{} for _ in configs]
        groups: Dict[Tuple, List[int]] = {}
        for i, cfg in enumerate(configs):
            groups.setdefault(structure_key(cfg), []).append(i)
        for indices in groups.values():
            try:
                wrapper = FaissIndexWrapper(configs[indices[0]], evaluator.dimension, evaluator.metric)
                build_time = wrapper.build_and_add(evaluator.database)
            except Exception as exc:
                for i in indices:
                    results[i] = {"error": f"build failed: {type(exc).__name__}: {exc}"}
                continue
            for i in indices:
                cfg = configs[i]
                try:
                    if cfg.family == IndexFamily.HNSW:
                        wrapper.set_runtime_parameters(efSearch=cfg.efSearch)
                    elif cfg.family in (IndexFamily.IVF_FLAT, IndexFamily.IVF_PQ):
                        wrapper.set_runtime_parameters(nprobe=cfg.nprobe)
                    passes = [evaluator.measure_index(wrapper, build_time, round_results=False)
                              for _ in range(max(1, latency_passes))]
                    r, lat = passes[0], summarize_latency_passes(passes)
                    results[i] = {"measured": {
                        "recall_at_k": r.recall_at_k,  # deterministic: identical in every pass
                        "latency_p50_ms": lat["latency_p50_ms"], "latency_p95_ms": lat["latency_p95_ms"],
                        "latency_p99_ms": lat["latency_p99_ms"], "latency_p95_cv": lat["latency_p95_cv"],
                        "latency_passes": lat["latency_repeats"],
                        "throughput_qps_sequential": lat["throughput_qps_sequential"], "build_time_s": r.index_build_time_s,
                        "serialized_index_mb": float(r.index_serialized_mb),
                        "estimated_memory_mb": r.memory_footprint_mb,
                        "effective_config": r.config.id_string,
                    }}
                except Exception as exc:
                    results[i] = {"error": f"measurement failed: {type(exc).__name__}: {exc}"}
            del wrapper
        return results

    @staticmethod
    def _measured_checks(m: Dict[str, Any], workload: WorkloadProfile) -> Dict[str, bool]:
        return {"recall": m["recall_at_k"] >= workload.target_recall,
                "memory": m["serialized_index_mb"] <= workload.memory_budget_mb,
                "latency": m["latency_p95_ms"] <= workload.max_latency_ms}

    # ------------------------------------------------------------------ main entry
    def tune(self, workload: WorkloadProfile, evaluator: Optional[BenchmarkEvaluator] = None) -> Dict[str, Any]:
        start = time.perf_counter()
        if evaluator is None:
            return self._prediction_only(workload, "no dataset available to benchmark the shortlist")

        candidates = self.synthetic.generate_candidate_space(workload, include_flat=True)
        preds, model_note = self.predict_candidates(workload, candidates)
        perf = [self._as_performance_prediction(p, workload) for p in preds]
        pareto = set(self.synthetic.compute_pareto_frontier(candidates, perf))

        shortlist, excluded = shortlist_candidates(preds, workload, self.shortlist_size,
                                                   families=[c.family.value for c in candidates],
                                                   max_per_family=self.max_per_family)
        roles = {i: "shortlist" for i in shortlist}

        t_bench = time.perf_counter()
        bench = dict(zip(shortlist, self.benchmark(evaluator, [candidates[i] for i in shortlist],
                                                   self.latency_passes)))

        # Safety net: exact Flat (recall 1.0) if nothing measured so far meets the recall target
        flat_idx = next((i for i, c in enumerate(candidates) if c.family == IndexFamily.FLAT), None)
        if (flat_idx is not None and flat_idx not in bench
                and not any("measured" in b and b["measured"]["recall_at_k"] >= workload.target_recall
                            for b in bench.values())
                and preds[flat_idx]["memory_mb"] <= workload.memory_budget_mb):
            bench[flat_idx] = self.benchmark(evaluator, [candidates[flat_idx]], self.latency_passes)[0]
            roles[flat_idx] = "safety_net"
        benchmark_ms = (time.perf_counter() - t_bench) * 1000.0

        measured_ok = {i: b["measured"] for i, b in bench.items() if "measured" in b}
        if not measured_ok:
            errors = "; ".join(sorted({b.get("error", "") for b in bench.values()}))
            result = self._prediction_only(workload, f"every shortlisted benchmark failed ({errors})")
            result["benchmark_errors"] = {candidates[i].id_string: b.get("error") for i, b in bench.items()}
            return result

        checks = {i: self._measured_checks(m, workload) for i, m in measured_ok.items()}
        feasible = [i for i in measured_ok if checks[i]["recall"] and checks[i]["memory"]]
        winner, tie_break = select_measured_winner(measured_ok, checks, self.tie_tolerance)
        wm, wc = measured_ok[winner], checks[winner]
        sla_satisfied = all(wc.values())
        violated = [name for name, key in (("recall", "recall"), ("latency_p95_ms", "latency"),
                                           ("memory_mb", "memory")) if not wc[key]]
        constraint_report = [
            {"constraint": "recall", "measured": wm["recall_at_k"], "predicted": preds[winner]["recall"],
             "target": workload.target_recall, "comparison": ">=", "met": wc["recall"], "basis": "measured"},
            {"constraint": "latency_p95_ms", "measured": wm["latency_p95_ms"],
             "predicted": preds[winner]["latency_p95_ms"], "target": workload.max_latency_ms,
             "comparison": "<=", "met": wc["latency"], "basis": "measured"},
            {"constraint": "memory_mb", "measured": wm["serialized_index_mb"], "predicted": preds[winner]["memory_mb"],
             "target": workload.memory_budget_mb, "comparison": "<=", "met": wc["memory"],
             "basis": "measured serialized index size (not process RAM)"},
        ]

        shortlist_report = []
        for i in list(bench):
            entry = {
                "config": candidates[i].to_dict(), "role": roles[i], "selected": i == winner,
                "predicted": {k: preds[i][k] for k in ("source", "recall", "latency_p50_ms", "latency_p95_ms",
                                                      "build_time_s", "memory_mb", "undertrained")},
                "measured": bench[i].get("measured"), "error": bench[i].get("error"),
            }
            if i in checks:
                entry["meets"] = checks[i]
            shortlist_report.append(entry)

        sources = {s: sum(p["source"] == s for p in preds) for s in (SOURCE_MEASURED, SOURCE_SYNTHETIC)}
        primary = SOURCE_MEASURED if sources[SOURCE_MEASURED] else SOURCE_SYNTHETIC
        rec_pred = perf[winner].to_dict()
        rec_pred["source"] = preds[winner]["source"]

        return {
            "tuning_mode": "predict_shortlist_benchmark",
            "selection_basis": "measured",
            "prediction_model": primary,
            "prediction_sources": sources,
            "fallback_reason": model_note,
            "recommended_config": candidates[winner].to_dict(),
            "recommended_prediction": rec_pred,
            "recommended_measured": wm,
            "sla_satisfied": sla_satisfied,
            "sla_status": "measured_feasible" if sla_satisfied else "measured_best_effort",
            "is_fallback": not sla_satisfied,
            "feasible_candidate_count": len([i for i in measured_ok if all(checks[i].values())]),
            "benchmarked_count": len(bench),
            "shortlist_size": len(shortlist),
            "constraint_report": constraint_report,
            "violated_constraints": violated,
            "shortlist": shortlist_report,
            "tie_break": tie_break,
            "excluded_counts": {reason: list(excluded.values()).count(reason) for reason in set(excluded.values())},
            "tuning_duration_ms": round((time.perf_counter() - start) * 1000.0, 2),
            "benchmark_duration_ms": round(benchmark_ms, 2),
            "total_candidates_evaluated": len(candidates),
            "pareto_optimal_count": len(pareto),
            "calibration_performed": False,
            "rationale": self._rationale(workload, candidates, preds, shortlist, bench, winner, wm, wc, feasible,
                                         primary, model_note, tie_break),
            "all_points": [
                {"id": c.id_string, "family": c.family.value, "recall": p["recall"],
                 "latency_p95": p["latency_p95_ms"], "memory_mb": p["memory_mb"], "source": p["source"],
                 "is_pareto": i in pareto, "is_feasible": perf[i].is_feasible, "is_winner": i == winner,
                 "is_shortlisted": i in bench,
                 "measured_recall": bench[i]["measured"]["recall_at_k"] if i in measured_ok else None,
                 "measured_latency_p95": bench[i]["measured"]["latency_p95_ms"] if i in measured_ok else None}
                for i, (c, p) in enumerate(zip(candidates, preds))
            ],
        }

    def _prediction_only(self, workload: WorkloadProfile, reason: str) -> Dict[str, Any]:
        """Original synthetic prediction-only optimizer (unchanged behaviour), clearly labelled."""
        result = self.synthetic.tune(workload)
        result.update({
            "tuning_mode": "prediction_only",
            "selection_basis": "predicted",
            "prediction_model": SOURCE_SYNTHETIC,
            "fallback_reason": reason,
            "recommended_measured": None,
            "shortlist": [],
            "benchmarked_count": 0,
        })
        result["rationale"] = f"[Prediction only: {reason}] " + result["rationale"]
        return result

    @staticmethod
    def _rationale(workload, candidates, preds, shortlist, bench, winner, wm, wc, feasible, primary, model_note,
                   tie_break=None) -> str:
        k = workload.target_k
        rec_v, rec_t = format_against_target(wm["recall_at_k"], workload.target_recall, _fmt_percent, 1)
        lat_v, lat_t = format_against_target(wm["latency_p95_ms"], workload.max_latency_ms, _fmt_ms, 3)
        mem_v, mem_t = format_against_target(wm["serialized_index_mb"], workload.memory_budget_mb, _fmt_mb, 1)
        model_txt = "the measured-data cost model" if primary == SOURCE_MEASURED else "the synthetic cost model (fallback)"
        head = (f"{model_txt.capitalize()} predicted {len(candidates)} configurations; {len(shortlist)} were "
                f"shortlisted and {len(bench)} benchmarked on the actual dataset. ")
        if model_note:
            head += f"Note: {model_note}. "
        body = (f"Selected {candidates[winner].id_string} from MEASURED results: recall@{k} {rec_v} "
                f"(target >= {rec_t}), p95 latency {lat_v} (limit {lat_t}), serialized index {mem_v} "
                f"(budget {mem_t}). ")
        if feasible and tie_break and tie_break.get("applied"):
            tail = (f"{len(tie_break['tied_configs'])} feasible candidates measured within "
                    f"{tie_break['tolerance']:.0%} of the fastest ({tie_break['fastest_config']}, "
                    f"{tie_break['fastest_p95_ms']:.3f} ms p95) - inside timing noise - so the one with the "
                    f"smallest index was chosen for a stable recommendation.")
            if not wc["latency"]:
                tail += " Its measured p95 latency exceeds the limit."
        elif feasible and all(wc.values()):
            tail = "It is the fastest benchmarked configuration meeting the recall and memory constraints."
        elif feasible:
            tail = ("It is the fastest benchmarked configuration meeting the recall and memory constraints, "
                    "but its measured p95 latency exceeds the limit.")
        else:
            tail = ("No benchmarked configuration met both the recall and memory constraints; this is a "
                    "best-effort choice (SLA NOT met).")
        return head + body + tail
