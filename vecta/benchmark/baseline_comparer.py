"""
VECTA Head-to-Head Baseline Comparison Engine.
Executes empirical comparison between Vecta Auto-Tuned configurations
and industry-standard FAISS default settings (HNSW and IVF-Flat).
"""

import numpy as np
from typing import Dict, Any, List, Optional
from vecta.config import (
    IndexConfig,
    IndexFamily,
    BenchmarkResult,
    WorkloadProfile,
    FAISS_DEFAULT_HNSW,
    FAISS_DEFAULT_IVF,
    MetricType
)
from vecta.benchmark.evaluator import BenchmarkEvaluator


class BaselineComparer:
    """Executes multi-configuration empirical arena benchmarks."""

    @staticmethod
    def describe_latency_ratio(speedup_factor: float) -> str:
        """Human-readable latency comparison; a factor < 1 is a slowdown, not a 'speedup'."""
        if speedup_factor >= 1.0:
            return f"{speedup_factor:.2f}x faster"
        return f"{1.0 / max(1e-9, speedup_factor):.2f}x slower"

    @staticmethod
    def check_benchmark_sla(result: BenchmarkResult, workload: WorkloadProfile) -> Dict[str, Any]:
        """
        Compare benchmark results against the workload targets. Recall and p95 latency are
        measured; index memory is an ESTIMATE (max of serialized index size and an analytical
        model), so each check states its `basis`.
        """
        checks = {
            "recall": {
                "value": result.recall_at_k,
                "basis": "measured",
                "target": workload.target_recall,
                "comparison": ">=",
                "met": result.recall_at_k >= workload.target_recall
            },
            "latency_p95_ms": {
                "value": result.latency_p95_ms,
                "basis": "measured",
                "target": workload.max_latency_ms,
                "comparison": "<=",
                "met": result.latency_p95_ms <= workload.max_latency_ms
            },
            "memory_mb": {
                "value": result.memory_footprint_mb,
                "basis": "estimated",
                "serialized_index_mb": result.index_serialized_mb,
                "target": workload.memory_budget_mb,
                "comparison": "<=",
                "met": result.memory_footprint_mb <= workload.memory_budget_mb
            }
        }
        checks["all_met"] = all(c["met"] for c in checks.values())
        return checks

    @staticmethod
    def run_head_to_head_arena(
        evaluator: BenchmarkEvaluator,
        vecta_config: IndexConfig,
        workload: Optional[WorkloadProfile] = None
    ) -> Dict[str, Any]:
        """
        Evaluate exact Flat search, FAISS Default HNSW, FAISS Default IVF, and Vecta Auto-Tuned.
        Returns comprehensive performance deltas. If `workload` is given, the Vecta result is also
        checked against its recall / latency (measured) and memory (estimated) targets.
        """
        # Run benchmarks: recall and latency are measured for every row; memory is an estimate
        res_flat = evaluator.evaluate_config(IndexConfig(family=IndexFamily.FLAT))
        res_default_hnsw = evaluator.evaluate_config(FAISS_DEFAULT_HNSW)
        res_default_ivf = evaluator.evaluate_config(FAISS_DEFAULT_IVF)
        res_vecta = evaluator.evaluate_config(vecta_config)

        # Determine reference default (best performing default by recall)
        if res_default_hnsw.recall_at_k >= res_default_ivf.recall_at_k:
            ref_default, ref_name = res_default_hnsw, "FAISS Default HNSW"
        else:
            ref_default, ref_name = res_default_ivf, "FAISS Default IVF"

        # Calculate comparative advantage (speedup_factor < 1 means Vecta is slower)
        speedup_factor = ref_default.latency_p95_ms / max(1e-4, res_vecta.latency_p95_ms)
        latency_ratio = res_vecta.latency_p95_ms / max(1e-4, ref_default.latency_p95_ms)
        recall_gain_abs = (res_vecta.recall_at_k - ref_default.recall_at_k) * 100.0

        mem_delta_pct = 0.0
        if ref_default.memory_footprint_mb > 0:
            mem_delta_pct = ((ref_default.memory_footprint_mb - res_vecta.memory_footprint_mb) / ref_default.memory_footprint_mb) * 100.0

        # Relative summary statement (recall / latency measured; memory estimated)
        summary_verdict = (
            f"Vecta Auto-Tuned configuration ({res_vecta.config.id_string}) measured "
            f"{res_vecta.recall_at_k * 100:.1f}% Recall@{evaluator.k} with {res_vecta.latency_p95_ms:.3f} ms p95 latency. "
            f"Compared to {ref_name} (the higher-recall default), its p95 latency is "
            f"{BaselineComparer.describe_latency_ratio(speedup_factor)}, recall is "
            f"{'+' if recall_gain_abs >= 0 else ''}{recall_gain_abs:.1f} points, and its estimated index memory is "
            f"{abs(mem_delta_pct):.1f}% {'lower' if mem_delta_pct >= 0 else 'higher'}."
        )

        sla_check = None
        if workload is not None:
            sla_check = BaselineComparer.check_benchmark_sla(res_vecta, workload)
            parts = []
            for key, label, unit_fmt in [
                ("recall", "measured recall", lambda v: f"{v * 100:.1f}%"),
                ("latency_p95_ms", "measured p95 latency", lambda v: f"{v:.3g} ms"),
                ("memory_mb", "estimated index memory", lambda v: f"{v:.1f} MB"),
            ]:
                c = sla_check[key]
                parts.append(
                    f"{label} {unit_fmt(c['value'])} vs {c['comparison']} {unit_fmt(c['target'])} "
                    f"({'met' if c['met'] else 'MISSED'})"
                )
            summary_verdict += (
                f" Against the workload targets: {'; '.join(parts)}. "
                + ("All targets met." if sla_check["all_met"] else "Not all targets were met.")
            )

        return {
            "summary_verdict": summary_verdict,
            "reference_default": ref_name,
            "speedup_factor": round(speedup_factor, 2),
            "latency_ratio_vs_default": round(latency_ratio, 2),
            "latency_comparison": BaselineComparer.describe_latency_ratio(speedup_factor),
            "recall_gain_abs_pct": round(recall_gain_abs, 2),
            "memory_reduction_pct": round(mem_delta_pct, 2),
            "sla_check": sla_check,
            "results": {
                "vecta_auto_tuned": res_vecta.to_dict(),
                "faiss_default_hnsw": res_default_hnsw.to_dict(),
                "faiss_default_ivf": res_default_ivf.to_dict(),
                "ground_truth_flat": res_flat.to_dict()
            }
        }
