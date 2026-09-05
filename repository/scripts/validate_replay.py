import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_experiments/analysis"))
import independent_metrics as independent


def main():
    parser = argparse.ArgumentParser(description="Independently recompute the numerical results and compare frozen tables.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"checks": {}, "scope": "stored predictions; no training or raw-object reconstruction"}
    data = args.data_root
    main_root = data / "04_analysis/reviewer_v13_20260904"
    metadata_root = data / "04_analysis/reviewer_v14_20260905/metadata_ablation"
    labels = independent.load_labels(data / "03_data_interaction/model_input/gate1_v2_20260821_0656/labels")
    paired = independent.parse_paired(data / "03_data_interaction/data_integrity/paired_label_contracts/paired_label_contracts.csv")
    original = pd.read_csv(main_root / "core/blind_predictions.csv")
    combined = pd.read_csv(args.results / "metadata/combined_blind_predictions.csv", float_precision="round_trip")
    keys = ["model", "seed", "fit_contract", "split", "file_name"]
    deep = ("cpmlp", "cptransformer")
    same_deep = original[original.model.isin(deep)].sort_values(keys).reset_index(drop=True).equals(
        combined[combined.model.isin(deep)].sort_values(keys).reset_index(drop=True))
    independent.add_check(report, "metadata_deep_predictions_reused_exactly", same_deep)
    independent.add_check(report, "prediction_counts_and_unique_keys",
                          len(original) == len(combined) == 58880 and not original.duplicated(keys).any()
                          and not combined.duplicated(keys).any() and np.isfinite(combined.predicted_eol).all())

    for label, predictions, frozen in (("main", original, main_root / "analysis"),
                                        ("metadata", combined, metadata_root)):
        print(f"Independent numerical checks: {label}", flush=True)
        output = args.results / label
        for path in sorted(output.glob("*.csv")):
            try:
                pd.testing.assert_frame_equal(pd.read_csv(path), pd.read_csv(frozen / path.name),
                                              check_exact=False, rtol=1e-10, atol=1e-10)
                independent.add_check(report, f"{label}/frozen/{path.name}", True)
            except AssertionError as error:
                independent.add_check(report, f"{label}/frozen/{path.name}", False, error=str(error))

        def compare(name, frame, filename, key):
            independent.compare_frame(report, f"{label}/independent/{name}", frame, output / filename, key)

        points, factors = independent.point_tables(predictions, labels)
        compare("point_metrics", points, "point_metrics_by_seed.csv", ["fit_contract", "evaluation_contract", "model", "seed"])
        compare("factor_contrasts", factors, "factor_contrasts_by_seed.csv", ["model", "seed", "metric"])
        ranks = independent.rank_table(points)
        compare("ranks", ranks, "model_ranks_by_cell.csv", ["cell", "seed", "metric", "model"])
        compare("winner_frequency", independent.best_frequency(ranks), "best_model_frequency.csv", ["cell", "metric", "model"])
        wide, cross, within, variability = independent.variability_tables(predictions, paired)
        compare("prediction_shifts", wide, "prediction_shift_cells.csv", ["file_name", "model", "seed"])
        compare("cross_contract_shifts", cross, "cross_contract_prediction_shift_by_seed.csv", ["model", "seed", "label_status"])
        compare("within_contract_refits", within, "within_contract_refit_pairwise.csv", ["model", "fit_contract", "seed_left", "seed_right", "label_status"])
        compare("refit_summary", variability, "refit_variability_summary.csv", ["model", "label_status"])
        replicates, summary = independent.shift_bootstrap(wide, 5000)
        compare("shift_bootstrap_draws", replicates, "prediction_shift_bootstrap_replicates.csv", ["replicate", "model", "label_status"])
        compare("shift_bootstrap_summary", summary, "prediction_shift_bootstrap_summary.csv", ["model", "label_status", "quantity"])
        if label == "main":
            replicates, summary = independent.factor_bootstrap(predictions, labels, 5000)
            compare("factor_bootstrap_draws", replicates, "factor_seed_cell_bootstrap_replicates.csv", ["replicate", "model", "metric"])
            compare("factor_bootstrap_summary", summary, "factor_seed_cell_bootstrap_summary.csv", ["model", "metric", "contrast"])
            conformal, trace, differences, summary = independent.conformal_tables(predictions, labels, paired)
            compare("conformal", conformal, "conformal_metrics.csv", ["contract", "model", "seed", "nominal", "calibration_ablation", "population"])
            compare("calibration_order_statistics", trace, "calibration_quantile_neighbourhood.csv", ["contract", "model", "seed", "nominal", "rank"])
            compare("conformal_paired_differences", differences, "conformal_paired_differences_by_seed.csv", ["model", "seed", "nominal"])
            compare("conformal_summary", summary, "conformal_paired_summary.csv", ["model", "nominal", "quantity"])
    report["passed_checks"] = sum(c["passed"] for c in report["checks"].values())
    report["total_checks"] = len(report["checks"])
    report["status"] = "PASS" if report["passed_checks"] == report["total_checks"] else "FAIL"
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
