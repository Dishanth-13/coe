"""
VECTA Multi-Objective Adaptive Optimization Engine.
Performs constrained Pareto frontier discovery, utility ranking,
and micro-sample active calibration across the ANN hyper-parameter design space.
"""

import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from vecta.config import (
    WorkloadProfile,
    IndexConfig,
    IndexFamily,
    PerformancePrediction,
    INDEX_SEARCH_SPACE,
    FAISS_DEFAULT_HNSW,
    FAISS_DEFAULT_IVF,
    max_ivf_nlist,
    largest_valid_pq_m
)
from vecta.cost_model.surrogate_model import SurrogateCostModel
from vecta.engine.faiss_wrapper import FaissIndexWrapper


def _fmt_percent(x: float, precision: int) -> str:
    return f"{x * 100:.{precision}f}%"


def _fmt_ms(x: float, precision: int) -> str:
    return f"{x:.{precision}g} ms"


def _fmt_mb(x: float, precision: int) -> str:
    return f"{x:.{precision}f} MB"


def format_against_target(value: float, target: float, fmt, precision: int, max_precision: int = 12):
    """
    Format a predicted value and its target for display. Precision is increased until the two
    strings differ whenever the underlying numbers differ, so rounding can never make a missed
    target look met (e.g. 0.494197 vs 0.4942 both printing as "49.4%").
    """
    value_txt, target_txt = fmt(value, precision), fmt(target, precision)
    while value != target and value_txt == target_txt and precision < max_precision:
        precision += 1
        value_txt, target_txt = fmt(value, precision), fmt(target, precision)
    return value_txt, target_txt


class AutoTuningOptimizer:
    """Adaptive search engine optimizing ANN parameters under strict SLA constraints."""

    def __init__(self, cost_model: Optional[SurrogateCostModel] = None):
        self.cost_model = cost_model or SurrogateCostModel()

    def generate_candidate_space(self, workload: WorkloadProfile, include_flat: bool = False) -> List[IndexConfig]:
        """Synthesize candidate configuration lattice across index families."""
        candidates: List[IndexConfig] = []
        d = workload.dimension
        
        # 1. HNSW Grid
        hnsw_space = INDEX_SEARCH_SPACE[IndexFamily.HNSW]
        for M in hnsw_space["M"]:
            for efC in hnsw_space["efConstruction"]:
                for efS in hnsw_space["efSearch"]:
                    candidates.append(IndexConfig(
                        family=IndexFamily.HNSW,
                        M=M,
                        efConstruction=efC,
                        efSearch=efS
                    ))
                    
        # IVF lists must be trainable on this dataset: skip nlist values FAISS would clamp, so every
        # candidate's predicted configuration is exactly the configuration that gets built.
        nlist_cap = max_ivf_nlist(workload.num_vectors)

        def buildable_nlists(values: List[int]) -> List[int]:
            valid = [v for v in values if v <= nlist_cap]
            return valid if valid else [nlist_cap]

        # 2. IVF-Flat Grid
        ivf_space = INDEX_SEARCH_SPACE[IndexFamily.IVF_FLAT]
        for nlist in buildable_nlists(ivf_space["nlist"]):
            for nprobe in ivf_space["nprobe"]:
                if nprobe <= nlist:
                    candidates.append(IndexConfig(
                        family=IndexFamily.IVF_FLAT,
                        nlist=nlist,
                        nprobe=nprobe
                    ))

        # 3. IVF-PQ Grid (memory-efficient)
        ivf_pq_space = INDEX_SEARCH_SPACE[IndexFamily.IVF_PQ]
        # Sub-quantizer counts come from the configured search space, restricted to divisors of D
        configured_m = sorted(set(int(m) for m in ivf_pq_space["m_pq_divisors"]), reverse=True)
        valid_divisors = [m for m in configured_m if 1 <= m <= d and d % m == 0]
        if not valid_divisors:
            # None of the configured values divides D: use the closest valid value below the largest
            valid_divisors = [largest_valid_pq_m(d, max(configured_m) if configured_m else d)]

        for nlist in buildable_nlists(ivf_pq_space["nlist"]):
            for nprobe in ivf_pq_space["nprobe"]:
                if nprobe <= nlist:
                    for m_pq in valid_divisors:
                        candidates.append(IndexConfig(
                            family=IndexFamily.IVF_PQ,
                            nlist=nlist,
                            nprobe=nprobe,
                            m_pq=m_pq
                        ))

        # 4. Exact Flat Baseline (optional, only if requested)
        if include_flat:
            candidates.append(IndexConfig(family=IndexFamily.FLAT))
        
        return candidates

    def compute_pareto_frontier(
        self,
        candidates: List[IndexConfig],
        predictions: List[PerformancePrediction]
    ) -> List[int]:
        """
        Identify indices of non-dominated solutions on the 3D objective manifold:
        Maximize Recall, Minimize Latency, Minimize Memory Footprint.
        """
        pareto_indices = []
        n = len(candidates)
        
        # Matrix: [Recall, -Latency, -Memory] so that all objectives are to be MAXIMIZED
        objectives = np.zeros((n, 3))
        for i, pred in enumerate(predictions):
            objectives[i, 0] = pred.predicted_recall
            objectives[i, 1] = -pred.predicted_latency_p95
            objectives[i, 2] = -pred.predicted_memory_mb
            
        for i in range(n):
            dominated = False
            for j in range(n):
                if i != j:
                    # j dominates i if j is >= in all objectives and strictly > in at least one
                    if (objectives[j, :] >= objectives[i, :]).all() and (objectives[j, :] > objectives[i, :]).any():
                        dominated = True
                        break
            if not dominated:
                pareto_indices.append(i)
                
        return pareto_indices

    def evaluate_utility(
        self,
        pred: PerformancePrediction,
        workload: WorkloadProfile
    ) -> float:
        """
        Multi-objective constrained utility function.
        Higher score = superior operational configuration.
        """
        # Recall utility: must achieve target SLA before latency optimization matters
        if pred.predicted_recall >= workload.target_recall:
            recall_score = 2.0 + 1.0 * (pred.predicted_recall - workload.target_recall)
        else:
            # Harsh penalty for failing recall SLA
            recall_score = -20.0 * (workload.target_recall - pred.predicted_recall)
            
        # Latency utility: normalized relative to max tolerable latency SLA
        lat_ratio = pred.predicted_latency_p95 / max(0.01, workload.max_latency_ms)
        if lat_ratio <= 1.0:
            latency_score = 1.0 - 0.4 * lat_ratio  # faster is better
        else:
            latency_score = -10.0 * (lat_ratio - 1.0)
            
        # Memory utility: normalized to budget
        mem_ratio = pred.predicted_memory_mb / max(1.0, workload.memory_budget_mb)
        if mem_ratio <= 1.0:
            memory_score = 1.0 - 0.2 * mem_ratio
        else:
            memory_score = -10.0 * (mem_ratio - 1.0)

        # Composite utility score with constraint penalty
        total_utility = (0.60 * recall_score) + (0.30 * latency_score) + (0.10 * memory_score)
        total_utility -= pred.constraint_violation_penalty
        
        return float(total_utility)

    @staticmethod
    def constraint_violation(pred: PerformancePrediction, workload: WorkloadProfile) -> float:
        """
        Total relative shortfall against the workload targets (0.0 when every target is met).
        Used to pick the closest candidate when no configuration is predicted to be feasible.
        """
        recall_gap = max(0.0, workload.target_recall - pred.predicted_recall) / max(1e-9, workload.target_recall)
        latency_gap = max(0.0, pred.predicted_latency_p95 - workload.max_latency_ms) / max(1e-9, workload.max_latency_ms)
        memory_gap = max(0.0, pred.predicted_memory_mb - workload.memory_budget_mb) / max(1e-9, workload.memory_budget_mb)
        return float(recall_gap + latency_gap + memory_gap)

    @staticmethod
    def describe_constraints(pred: PerformancePrediction, workload: WorkloadProfile) -> List[Dict[str, Any]]:
        """Per-target comparison of a prediction against the workload SLAs."""
        return [
            {
                "constraint": "recall",
                "predicted": pred.predicted_recall,
                "target": workload.target_recall,
                "comparison": ">=",
                "met": bool(pred.meets_recall_target)
            },
            {
                "constraint": "latency_p95_ms",
                "predicted": pred.predicted_latency_p95,
                "target": workload.max_latency_ms,
                "comparison": "<=",
                "met": bool(pred.meets_latency_target)
            },
            {
                "constraint": "memory_mb",
                "predicted": pred.predicted_memory_mb,
                "target": workload.memory_budget_mb,
                "comparison": "<=",
                "met": bool(pred.meets_memory_budget)
            }
        ]

    @staticmethod
    def _build_rationale(
        cfg: IndexConfig,
        pred: PerformancePrediction,
        workload: WorkloadProfile,
        feasible_count: int,
        total_count: int
    ) -> str:
        k = workload.target_k
        rec_v, rec_t = format_against_target(pred.predicted_recall, workload.target_recall, _fmt_percent, 1)
        lat_v, lat_t = format_against_target(pred.predicted_latency_p95, workload.max_latency_ms, _fmt_ms, 3)
        mem_v, mem_t = format_against_target(pred.predicted_memory_mb, workload.memory_budget_mb, _fmt_mb, 1)
        recall_txt = f"{rec_v} Recall@{k} (target >= {rec_t})"
        latency_txt = f"{lat_v} p95 latency (limit {lat_t})"
        memory_txt = f"{mem_v} memory (budget {mem_t})"

        if feasible_count > 0:
            return (
                f"Vecta selected {cfg.id_string}, the highest-utility of {feasible_count} of {total_count} "
                f"candidate configurations predicted to meet every target: predicted {recall_txt}, "
                f"{latency_txt} and {memory_txt}. These are cost-model predictions, not measurements; "
                f"run the benchmark to verify them."
            )

        misses = []
        if not pred.meets_recall_target:
            misses.append(f"recall {rec_v} vs target >= {rec_t}")
        if not pred.meets_latency_target:
            misses.append(f"p95 latency {lat_v} vs limit {lat_t}")
        if not pred.meets_memory_budget:
            misses.append(f"memory {mem_v} vs budget {mem_t}")
        return (
            f"No configuration in the {total_count}-candidate search space is predicted to meet all targets. "
            f"Falling back to the closest candidate, {cfg.id_string} (best-effort, SLA NOT met): it is predicted "
            f"to miss {'; '.join(misses)}. Relax the targets or treat this recommendation as best-effort."
        )

    def tune(
        self,
        workload: WorkloadProfile,
        active_calibration_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Run full auto-tuning pipeline:
        1. Generate search grid
        2. Surrogate cost model prediction
        3. Pareto frontier extraction
        4. Constrained ranking & winner selection
        5. (Optional) Micro-sample empirical verification
        """
        start_time = time.perf_counter()
        
        # 1. Candidate space
        candidates = self.generate_candidate_space(workload)
        
        # 2. ML Surrogate Predictions
        predictions = self.cost_model.predict_batch(workload, candidates)
        
        # 3. Pareto Frontier Discovery
        pareto_indices = self.compute_pareto_frontier(candidates, predictions)
        
        # 4. Rank Candidates: configurations predicted to meet every target come first (by utility);
        #    if none exist, rank by smallest total constraint violation (honest best-effort fallback).
        pareto_set = set(pareto_indices)
        scored_candidates = []
        for i, (cfg, pred) in enumerate(zip(candidates, predictions)):
            score = self.evaluate_utility(pred, workload)
            scored_candidates.append({
                "index": i,
                "config": cfg,
                "prediction": pred,
                "score": score,
                "violation": self.constraint_violation(pred, workload),
                "is_pareto": i in pareto_set
            })

        feasible_count = sum(1 for item in scored_candidates if item["prediction"].is_feasible)
        if feasible_count > 0:
            scored_candidates.sort(key=lambda item: (not item["prediction"].is_feasible, -item["score"]))
        else:
            scored_candidates.sort(key=lambda item: (item["violation"], -item["score"]))
        winner_entry = scored_candidates[0]
        winner_index = winner_entry["index"]
        winner_cfg = winner_entry["config"]
        winner_pred = winner_entry["prediction"]
        sla_satisfied = bool(winner_pred.is_feasible)
        
        # Optional: Micro-sample active verification
        calibration_performed = False
        if active_calibration_data is not None:
            vectors_sample, queries_sample = active_calibration_data
            if len(vectors_sample) >= 50 and len(queries_sample) >= 5:
                # Fast micro-benchmark on winning candidate
                try:
                    micro_wrapper = FaissIndexWrapper(winner_cfg, workload.dimension, workload.metric)
                    micro_wrapper.build_and_add(vectors_sample)
                    _, _, micro_lat = micro_wrapper.search(queries_sample, k=workload.target_k)
                    
                    # Calibrate cost model
                    self.cost_model.calibrate([
                        (winner_cfg, winner_pred.predicted_recall, micro_lat / len(queries_sample), winner_pred.predicted_memory_mb)
                    ])
                    calibration_performed = True
                except Exception as e:
                    pass

        tuning_duration_ms = (time.perf_counter() - start_time) * 1000.0

        # Rationale generation: states whether the targets are predicted to be met, never assumes it
        rationale = self._build_rationale(winner_cfg, winner_pred, workload, feasible_count, len(candidates))
        constraint_report = self.describe_constraints(winner_pred, workload)

        return {
            "recommended_config": winner_cfg.to_dict(),
            "recommended_prediction": winner_pred.to_dict(),
            "winner_score": round(winner_entry["score"], 4),
            "sla_satisfied": sla_satisfied,
            "sla_status": "predicted_feasible" if sla_satisfied else "no_feasible_configuration",
            "is_fallback": not sla_satisfied,
            "feasible_candidate_count": feasible_count,
            "constraint_report": constraint_report,
            "violated_constraints": [c["constraint"] for c in constraint_report if not c["met"]],
            "tuning_duration_ms": round(tuning_duration_ms, 2),
            "total_candidates_evaluated": len(candidates),
            "pareto_optimal_count": len(pareto_indices),
            "calibration_performed": calibration_performed,
            "rationale": rationale,
            "top_candidates": [
                {
                    "config": item["config"].to_dict(),
                    "prediction": item["prediction"].to_dict(),
                    "score": round(item["score"], 4),
                    "is_pareto": item["is_pareto"],
                    "is_feasible": item["prediction"].is_feasible
                }
                for item in scored_candidates[:8]
            ],
            "all_points": [
                {
                    "id": cfg.id_string,
                    "family": cfg.family.value,
                    "recall": pred.predicted_recall,
                    "latency_p95": pred.predicted_latency_p95,
                    "memory_mb": pred.predicted_memory_mb,
                    "is_pareto": i in pareto_set,
                    "is_feasible": pred.is_feasible,
                    "is_winner": i == winner_index
                }
                for i, (cfg, pred) in enumerate(zip(candidates, predictions))
            ]
        }
