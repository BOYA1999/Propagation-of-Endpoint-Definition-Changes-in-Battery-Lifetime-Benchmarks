from pathlib import Path
import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics


parser = argparse.ArgumentParser(description="Recalculate matched-contract lower-rank reversals from stored scores.")
parser.add_argument("--data-root", type=Path, required=True)
parser.add_argument("--analysis-root", type=Path, help="Use freshly recomputed controlled-analysis scores.")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
ROOT = args.data_root.resolve()
SOURCE = args.analysis_root.resolve() if args.analysis_root else ROOT / "04_analysis/reviewer_v13_20260904/analysis"
OUT = args.output.resolve()
if OUT.is_relative_to(ROOT) or (OUT.exists() and any(OUT.iterdir())):
    parser.error("--output must be a new or empty directory outside --data-root")
METRICS = ["mape", "smape", "mae", "rmse", "log_mae"]
FILES = ["point_metrics_by_seed.csv", "model_ranks_by_cell.csv", "ranking_comparisons.csv"]


def read_csv(name):
    with (SOURCE / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(name, rows):
    with (OUT / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


point_rows = [r for r in read_csv(FILES[0]) if r["fit_contract"] == r["evaluation_contract"]]
scores = {
    ("AA" if r["fit_contract"] == "release" else "BB", int(r["seed"]), r["model"]):
    {metric: float(r[metric]) for metric in METRICS}
    for r in point_rows
}
seeds = sorted({key[1] for key in scores})
models = sorted({key[2] for key in scores})
rank_rows = [r for r in read_csv(FILES[1]) if r["cell"] in ["AA", "BB"]]
comparisons = [r for r in read_csv(FILES[2]) if r["direction"] == "matched_contract"]
orders, events, ties = {}, [], 0
for seed, metric in itertools.product(seeds, METRICS):
    orders[(seed, metric)] = {
        cell: sorted(models, key=lambda model: scores[(cell, seed, model)][metric])
        for cell in ["AA", "BB"]
    }
    for model_i, model_j in itertools.combinations(models, 2):
        aa = [scores[("AA", seed, model)][metric] for model in [model_i, model_j]]
        bb = [scores[("BB", seed, model)][metric] for model in [model_i, model_j]]
        ties += int(aa[0] == aa[1] or bb[0] == bb[1])
        if (aa[0] - aa[1]) * (bb[0] - bb[1]) < 0:
            event = dict(seed=seed, metric=metric, model_i=model_i, model_j=model_j)
            for cell, values in [("AA", aa), ("BB", bb)]:
                event.update({
                    f"score_i_{cell}": values[0], f"score_j_{cell}": values[1],
                    f"absolute_score_gap_{cell}": abs(values[0] - values[1]),
                    f"relative_gap_pct_{cell}": 200 * abs(values[0] - values[1]) / sum(values),
                })
            events.append(event)

summary = []
for metric in METRICS:
    selected = [event for event in events if event["metric"] == metric]
    row = dict(metric=metric, changed_comparisons=len({r["seed"] for r in selected}),
               total_comparisons=len(seeds), pair_reversal_events=len(selected))
    for cell in ["AA", "BB"]:
        gaps = [r[f"relative_gap_pct_{cell}"] for r in selected]
        row.update({f"median_gap_pct_{cell}": statistics.median(gaps),
                    f"min_gap_pct_{cell}": min(gaps), f"max_gap_pct_{cell}": max(gaps)})
    summary.append(row)

changed = {key for key, value in orders.items() if value["AA"] != value["BB"]}
checks = {
    "matched_point_grid_184_unique": len(point_rows) == len(scores) == 184 and len(seeds) == 23 and len(models) == 4,
    "all_scores_finite_positive": all(math.isfinite(v) and v > 0 for row in scores.values() for v in row.values()),
    "matched_rank_grid_920_unique": len(rank_rows) == len({(r["cell"], r["seed"], r["model"], r["metric"]) for r in rank_rows}) == 920,
    "all_920_rank_scores_match_point_metrics": all(math.isclose(float(r["score"]), scores[(r["cell"], int(r["seed"]), r["model"])][r["metric"]], rel_tol=1e-13, abs_tol=0) for r in rank_rows),
    "all_115_orders_match_frozen_comparisons": len(comparisons) == len({(r["seed"], r["metric"]) for r in comparisons}) == 115 and all(r["order_left"].split(">") == orders[(int(r["seed"]), r["metric"])]["AA"] and r["order_right"].split(">") == orders[(int(r["seed"]), r["metric"])]["BB"] for r in comparisons),
    "full_order_flags_match": all((r["full_order_same"] == "True") == ((int(r["seed"]), r["metric"]) not in changed) for r in comparisons),
    "no_score_ties": ties == 0,
    "28_changed_comparisons_30_unique_pair_events": len(changed) == 28 and len(events) == len({(r["seed"], r["metric"], r["model_i"], r["model_j"]) for r in events}) == 30 and {(r["seed"], r["metric"]) for r in events} == changed,
    "winner_stable_all_115": all(value["AA"][0] == value["BB"][0] == "xgboost" for value in orders.values()),
    "all_reversals_lower_rank_pairs": all("xgboost" not in [r["model_i"], r["model_j"]] for r in events),
}
OUT.mkdir(parents=True, exist_ok=True)
write_csv("events.csv", events)
write_csv("summary.csv", summary)
(OUT / "README.md").write_text("""# Lower-rank score gaps in matched-contract comparisons

This analysis reconstructs AA (A fit, A evaluation) and BB (B fit, B evaluation) rankings from the stored point metrics.

The 115 seed-by-metric comparisons contain 28 full-order changes and 30 distinct seed-by-metric-by-model-pair reversal events. Twenty-six comparisons each contain one event; seed 16 contains two events for sMAPE and two for log-MAE. The three model-pair identities are CPTransformer–Ridge (15 events), CPMLP–CPTransformer (11), and CPMLP–Ridge (4). XGBoost remains first throughout.

For every reversing pair, the gap is calculated separately in AA and BB as 2 * abs(score_i - score_j) / (score_i + score_j), equivalent to the absolute score difference divided by the two-score mean. CSV relative_gap_pct fields multiply this quantity by 100. All scores are positive. Each event contributes two gap values, so there are 60 values but only 30 pair-reversal events within 28 comparisons.

summary.csv gives metric-specific AA and BB medians and observed minima/maxima. Raw MAE/RMSE differences in events.csv retain processed-cycle-index units; other raw differences retain their own metric units. No raw scores are pooled across metrics. These summaries condition on observed reversal events; they are descriptive, not confidence intervals, independent samples, or a test of practical equivalence. The maximum relative gap is 27.53%, so the reversals are not uniformly near ties.

Run the script with `--data-root DATA --output RESULTS/rank_gap`. To use recomputed scores, also supply `--analysis-root RESULTS/controlled`. Only the Python standard library is required. validation.json records source hashes and checks against all 920 matched-contract model-score records and all 115 stored orderings.
""", encoding="utf-8")
validation = {
    "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
    "counts": {"matched_comparisons": len(orders), "changed_comparisons": len(changed),
               "pair_reversal_events": len(events), "contract_specific_gap_values": 2 * len(events),
               "matched_rank_scores_checked": len(rank_rows)},
    "relative_gap_definition": "2*abs(score_i-score_j)/(score_i+score_j); CSV values reported as percent",
    "source_sha256": {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() for name in FILES},
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "output_sha256": {name: hashlib.sha256((OUT / name).read_bytes()).hexdigest() for name in ["events.csv", "summary.csv", "README.md"]},
}
(OUT / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status": validation["status"], "passed": sum(checks.values()), "total": len(checks), **validation["counts"]}))
raise SystemExit(0 if all(checks.values()) else 1)
