"""
VECTA Cost-Model Evaluation (measured data).

Leave-one-dataset-out (LODO) evaluation of MeasuredCostModel against the original synthetic-trained
SurrogateCostModel, using benchmark rows from `vecta.benchmark.collector`.

Leakage control: in each fold an ENTIRE dataset is held out, so no configuration, structural
index build or dataset statistic of the test dataset is seen during training.

Run:  python -m vecta.cost_model.evaluation [--data path/to/measurements.csv]
"""

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import kendalltau, spearmanr

from vecta.config import IndexFamily, MetricType, WorkloadProfile
from vecta.benchmark.collector import load_measurements
from vecta.cost_model.measured_model import MeasuredCostModel, config_from_row, is_undertrained, dataset_from_row
from vecta.cost_model.memory_estimator import estimate_index_memory_mb
from vecta.cost_model.surrogate_model import SurrogateCostModel

DEFAULT_RECALL_TARGETS = (0.90, 0.95, 0.99)
SHORTLIST_SIZE = 5
FAMILIES = ("FLAT", "HNSW", "IVF_FLAT", "IVF_PQ")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def regression_metrics(pred: Sequence[float], true: Sequence[float], ratio_metrics: bool = False) -> Dict[str, Any]:
    """MAE, RMSE, R^2 and relative errors. For positive quantities, ratio_metrics adds pred/true factors."""
    p, t = np.asarray(pred, float), np.asarray(true, float)
    if len(t) == 0:
        return {"n": 0}
    err = p - t
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    out = {
        "n": int(len(t)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1.0 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else None,
        "median_abs_pct_error": float(np.median(np.abs(err) / np.maximum(np.abs(t), 1e-12)) * 100.0),
    }
    if ratio_metrics:
        ratio = np.maximum(p, 1e-12) / np.maximum(t, 1e-12)
        factor = np.exp(np.abs(np.log(ratio)))
        out.update({
            "median_ratio_pred_over_true": float(np.median(ratio)),
            "median_error_factor": float(np.median(factor)),
            "within_25pct": float(np.mean(factor <= 1.25)),
            "within_2x": float(np.mean(factor <= 2.0)),
            "r2_log": (float(1.0 - np.sum(np.log(ratio) ** 2) / np.sum((np.log(t) - np.log(t).mean()) ** 2))
                       if np.sum((np.log(t) - np.log(t).mean()) ** 2) > 0 else None),
        })
    return out


def rank_metrics(pred: Sequence[float], true: Sequence[float]) -> Dict[str, Any]:
    p, t = np.asarray(pred, float), np.asarray(true, float)
    if len(t) < 3 or np.all(p == p[0]) or np.all(t == t[0]):
        return {"n": int(len(t)), "spearman": None, "kendall": None}
    return {"n": int(len(t)), "spearman": float(spearmanr(p, t)[0]), "kendall": float(kendalltau(p, t)[0])}


def selection_outcome(rows: List[Dict[str, Any]], pred_recall: Sequence[float], pred_p95: Sequence[float],
                      target: float, shortlist_size: int = SHORTLIST_SIZE) -> Dict[str, Any]:
    """
    How a model's predictions would steer configuration choice on one dataset:
      * direct pick: fastest configuration predicted to reach the recall target;
      * shortlist: top-K by prediction, then pick the best MEASURED feasible one within it
        (the planned predict -> shortlist -> benchmark workflow).
    Compared with the oracle: the fastest configuration whose MEASURED recall meets the target.
    """
    meas_r = np.array([r["recall_at_k"] for r in rows])
    meas_l = np.array([r["latency_p95_ms"] for r in rows])
    pr, pl = np.asarray(pred_recall, float), np.asarray(pred_p95, float)
    feasible = np.where(meas_r >= target)[0]
    if len(feasible) == 0:
        return {"target": target, "oracle": None}
    oracle = int(feasible[np.argmin(meas_l[feasible])])

    pred_ok = np.where(pr >= target)[0]
    # Ranking: predicted-feasible by predicted latency, then the rest by predicted recall (closest first)
    ordered = list(pred_ok[np.argsort(pl[pred_ok])]) + \
        [i for i in np.argsort(-pr) if i not in set(pred_ok)]
    direct = ordered[0]
    shortlist = ordered[:shortlist_size]
    short_feasible = [i for i in shortlist if meas_r[i] >= target]
    short_pick = min(short_feasible, key=lambda i: meas_l[i]) if short_feasible else None

    def describe(i):
        return {"config_id": rows[i]["config_id"], "measured_recall": float(meas_r[i]),
                "measured_p95_ms": float(meas_l[i]), "meets_target": bool(meas_r[i] >= target)}

    return {
        "target": target,
        "oracle": describe(oracle),
        "direct_pick": describe(direct),
        "direct_meets_target": bool(meas_r[direct] >= target),
        "direct_latency_regret": float(meas_l[direct] / meas_l[oracle]) if meas_r[direct] >= target else None,
        "shortlist": [rows[i]["config_id"] for i in shortlist],
        "shortlist_contains_oracle": oracle in shortlist,
        "shortlist_pick": describe(short_pick) if short_pick is not None else None,
        "shortlist_latency_regret": float(meas_l[short_pick] / meas_l[oracle]) if short_pick is not None else None,
    }


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

def surrogate_predictions(rows: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """Original synthetic-trained model (unchanged) on the same rows. It never saw any measured data."""
    model = SurrogateCostModel()
    out = []
    for r in rows:
        wp = WorkloadProfile(num_vectors=r["num_vectors"], dimension=r["dimension"], metric=MetricType(r["metric"]),
                             target_k=r["k"], intrinsic_dimension=r["profile_intrinsic_dimension"],
                             norm_variance=r["profile_norm_variance"],
                             clustering_tendency_hopkins=r["profile_hopkins"])
        p = model.predict(wp, config_from_row(r))
        out.append({"recall": p.predicted_recall, "p95": p.predicted_latency_p95,
                    "build": p.predicted_build_time_s, "memory": p.predicted_memory_mb})
    return out


def lodo_predictions(rows: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """Leave-one-dataset-out predictions from MeasuredCostModel, aligned with `rows`."""
    datasets = sorted({r["dataset_name"] for r in rows})
    if len(datasets) < 2:
        raise ValueError("leave-one-dataset-out needs at least two datasets")
    preds: List[Optional[Dict[str, float]]] = [None] * len(rows)
    for held_out in datasets:
        train = [r for r in rows if r["dataset_name"] != held_out]
        model = MeasuredCostModel().fit(train)
        for i, r in enumerate(rows):
            if r["dataset_name"] == held_out:
                p = model.predict_row(r)
                preds[i] = {"recall": p.predicted_recall, "p95": p.predicted_latency_p95_ms,
                            "build": p.predicted_build_time_s, "memory": p.estimated_memory_mb,
                            "undertrained": p.undertrained}
    return preds


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _metrics_block(rows, preds, key, true_key, ratio, select=None, unique_builds=False):
    idx = [i for i, r in enumerate(rows) if select is None or select(r)]
    if unique_builds:  # build time: one sample per build
        seen, keep = set(), []
        for i in idx:
            b = (rows[i]["structure_id"], rows[i]["repeat"])
            if b not in seen:
                seen.add(b)
                keep.append(i)
        idx = keep
    return regression_metrics([preds[i][key] for i in idx], [rows[i][true_key] for i in idx], ratio)


def evaluate(rows: List[Dict[str, Any]], recall_targets: Sequence[float] = DEFAULT_RECALL_TARGETS) -> Dict[str, Any]:
    rows = [r for r in rows if r["status"] == "ok"]
    new = lodo_predictions(rows)
    old = surrogate_predictions(rows)
    datasets = sorted({r["dataset_name"] for r in rows})

    report: Dict[str, Any] = {
        "protocol": ("leave-one-dataset-out: for each of the {n} datasets, MeasuredCostModel is trained on the "
                     "other {m} and evaluated on every configuration of the held-out dataset. The original "
                     "SurrogateCostModel (synthetic-trained) is evaluated on the same rows without retraining."
                     ).format(n=len(datasets), m=len(datasets) - 1),
        "rows": len(rows), "datasets": datasets,
        "accuracy": {}, "ranking": {}, "selection": {}, "memory": {}, "undertrained": {},
    }

    targets = [("recall", "recall_at_k", False, False), ("p95", "latency_p95_ms", True, False),
               ("build", "build_time_s", True, True)]
    for name, preds in (("measured_model", new), ("synthetic_model", old)):
        acc = {}
        for key, true_key, ratio, unique in targets:
            acc[key] = {"all": _metrics_block(rows, preds, key, true_key, ratio, unique_builds=unique)}
            for fam in FAMILIES:
                acc[key][fam] = _metrics_block(rows, preds, key, true_key, ratio,
                                               select=lambda r, f=fam: r["family"] == f, unique_builds=unique)
        report["accuracy"][name] = acc

        rank = {}
        for ds in datasets:
            idx = [i for i, r in enumerate(rows) if r["dataset_name"] == ds]
            entry = {
                "recall_all": rank_metrics([preds[i]["recall"] for i in idx], [rows[i]["recall_at_k"] for i in idx]),
                "p95_all": rank_metrics([preds[i]["p95"] for i in idx], [rows[i]["latency_p95_ms"] for i in idx]),
            }
            for fam in ("HNSW", "IVF_FLAT", "IVF_PQ"):
                fi = [i for i in idx if rows[i]["family"] == fam]
                entry[f"recall_{fam}"] = rank_metrics([preds[i]["recall"] for i in fi], [rows[i]["recall_at_k"] for i in fi])
                entry[f"p95_{fam}"] = rank_metrics([preds[i]["p95"] for i in fi], [rows[i]["latency_p95_ms"] for i in fi])
            rank[ds] = entry
        report["ranking"][name] = rank

        sel = {}
        for ds in datasets:
            idx = [i for i, r in enumerate(rows) if r["dataset_name"] == ds]
            sel[ds] = [selection_outcome([rows[i] for i in idx], [preds[i]["recall"] for i in idx],
                                         [preds[i]["p95"] for i in idx], t) for t in recall_targets]
        report["selection"][name] = sel

    # Memory: index-specific calculation vs MEASURED serialized size (one sample per build)
    report["memory"]["index_specific_estimate"] = {
        fam: _metrics_block(rows, new, "memory", "serialized_index_mb", True,
                            select=lambda r, f=fam: r["family"] == f, unique_builds=True) for fam in FAMILIES}
    report["memory"]["synthetic_model"] = {
        fam: _metrics_block(rows, old, "memory", "serialized_index_mb", True,
                            select=lambda r, f=fam: r["family"] == f, unique_builds=True) for fam in FAMILIES}

    # Under-trained configurations (FAISS k-means warnings): reported separately, never silently pooled
    flagged = [i for i, r in enumerate(rows) if new[i]["undertrained"]]
    warned = [i for i, r in enumerate(rows) if r.get("faiss_warnings")]
    report["undertrained"] = {
        "rows_flagged_by_model": len(flagged),
        "rows_with_recorded_faiss_warnings": len(warned),
        "flag_matches_recorded_warnings": sorted(flagged) == sorted(warned),
        "measured_model_on_undertrained": {
            "recall": regression_metrics([new[i]["recall"] for i in flagged], [rows[i]["recall_at_k"] for i in flagged]),
            "p95": regression_metrics([new[i]["p95"] for i in flagged], [rows[i]["latency_p95_ms"] for i in flagged], True),
        },
        "measured_model_on_normal_ivf_flat": {
            "recall": _metrics_block(rows, new, "recall", "recall_at_k", False,
                                     select=lambda r: r["family"] == "IVF_FLAT" and not r.get("faiss_warnings")),
        },
    }
    report["feature_importances_all_data"] = MeasuredCostModel().fit(rows).feature_importances()
    return report


def _fmt(v, spec=".3f"):
    return "n/a" if v is None else format(v, spec)


def summarize(report: Dict[str, Any]) -> str:
    lines = [report["protocol"], ""]
    for key, label in (("recall", "Recall@k"), ("p95", "p95 latency"), ("build", "Build time (per build)")):
        lines.append(f"== {label}")
        lines.append(f"  {'scope':9s} {'model':10s} {'n':>4s} {'MAE':>9s} {'RMSE':>9s} {'R2':>7s} {'med|%err|':>9s}"
                     + ("  med factor  <=25%  <=2x" if key != "recall" else ""))
        for scope in ("all",) + FAMILIES:
            for name, short in (("measured_model", "measured"), ("synthetic_model", "synthetic")):
                m = report["accuracy"][name][key][scope]
                if not m.get("n"):
                    continue
                line = (f"  {scope:9s} {short:10s} {m['n']:4d} {_fmt(m['mae'], '.4f'):>9s} {_fmt(m['rmse'], '.4f'):>9s} "
                        f"{_fmt(m['r2'], '.2f'):>7s} {_fmt(m['median_abs_pct_error'], '.1f'):>8s}%")
                if key != "recall":
                    line += (f"  {_fmt(m['median_error_factor'], '.2f'):>9s}x {m['within_25pct']:5.0%} {m['within_2x']:5.0%}")
                lines.append(line)
        lines.append("")

    lines.append("== Ranking within each held-out dataset (Spearman; all configs / per family)")
    for ds in report["datasets"]:
        for name, short in (("measured_model", "measured"), ("synthetic_model", "synthetic")):
            e = report["ranking"][name][ds]
            fams = " ".join(f"{f}:{_fmt(e[f'recall_{f}']['spearman'], '+.2f')}/{_fmt(e[f'p95_{f}']['spearman'], '+.2f')}"
                            for f in ("HNSW", "IVF_FLAT", "IVF_PQ"))
            lines.append(f"  {ds:18s} {short:9s} recall {_fmt(e['recall_all']['spearman'], '+.2f')}  "
                         f"p95 {_fmt(e['p95_all']['spearman'], '+.2f')}  | recall/p95 {fams}")
    lines.append("")

    lines.append(f"== Configuration choice (oracle = fastest config with MEASURED recall >= target; shortlist K={SHORTLIST_SIZE})")
    for name, short in (("measured_model", "measured"), ("synthetic_model", "synthetic")):
        sel = report["selection"][name]
        flat = [o for ds in report["datasets"] for o in sel[ds] if o.get("oracle")]
        direct_ok = np.mean([o["direct_meets_target"] for o in flat])
        regrets = [o["direct_latency_regret"] for o in flat if o["direct_latency_regret"] is not None]
        hit = np.mean([o["shortlist_contains_oracle"] for o in flat])
        short_ok = np.mean([o["shortlist_pick"] is not None for o in flat])
        s_reg = [o["shortlist_latency_regret"] for o in flat if o["shortlist_latency_regret"] is not None]
        brute = sum(o["direct_pick"]["config_id"].startswith("Flat") for o in flat)
        lines.append(f"  {short:9s} cases={len(flat)} | direct pick = brute-force Flat in {brute} | "
                     f"direct pick meets target {direct_ok:.0%}, median latency regret "
                     f"{_fmt(np.median(regrets) if regrets else None, '.2f')}x | shortlist has a feasible config "
                     f"{short_ok:.0%}, contains oracle {hit:.0%}, median regret {_fmt(np.median(s_reg) if s_reg else None, '.2f')}x")
    lines.append("")

    lines.append("== Memory vs MEASURED serialized index size (per build)")
    for fam in FAMILIES:
        a = report["memory"]["index_specific_estimate"][fam]
        b = report["memory"]["synthetic_model"][fam]
        lines.append(f"  {fam:9s} index-specific: median error factor {_fmt(a.get('median_error_factor'), '.4f')}x "
                     f"(MAE {_fmt(a.get('mae'), '.4f')} MB) | synthetic model: median factor "
                     f"{_fmt(b.get('median_error_factor'), '.2f')}x, within 25% {b.get('within_25pct', 0):.0%}")
    u = report["undertrained"]
    lines.append("")
    lines.append(f"== Under-trained IVF configurations: {u['rows_flagged_by_model']} flagged by model, "
                 f"{u['rows_with_recorded_faiss_warnings']} with recorded FAISS warnings, identical sets: "
                 f"{u['flag_matches_recorded_warnings']}")
    lines.append(f"  recall MAE on flagged rows {_fmt(u['measured_model_on_undertrained']['recall'].get('mae'), '.4f')}"
                 f" vs normal IVF-Flat rows {_fmt(u['measured_model_on_normal_ivf_flat']['recall'].get('mae'), '.4f')}")
    return "\n".join(lines)


def find_latest_measurements(results_dir: str = os.path.join("benchmarks", "results")) -> str:
    """Newest run whose manifest is complete and uses schema >= 2."""
    candidates = []
    for manifest in glob.glob(os.path.join(results_dir, "*", "run_metadata.json")):
        with open(manifest, encoding="utf-8") as f:
            m = json.load(f)
        if m.get("status") == "completed" and m.get("schema_version", 0) >= 2:
            candidates.append((m.get("started_utc", ""), os.path.join(os.path.dirname(manifest), "measurements.csv")))
    if not candidates:
        raise FileNotFoundError(f"no completed schema-v2 benchmark run under {results_dir}")
    return sorted(candidates)[-1][1]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the measured-data cost model (leave-one-dataset-out).")
    parser.add_argument("--data", help="measurements.csv (default: newest completed schema-v2 run)")
    parser.add_argument("--output", help="write the full JSON report here (default: next to the data as model_evaluation.json)")
    args = parser.parse_args(argv)
    data = args.data or find_latest_measurements()
    rows = load_measurements(data)
    report = evaluate(rows)
    report["data"] = data
    output = args.output or os.path.join(os.path.dirname(data), "model_evaluation.json")
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Data: {data}\n")
    print(summarize(report))
    print(f"\nFull report: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
