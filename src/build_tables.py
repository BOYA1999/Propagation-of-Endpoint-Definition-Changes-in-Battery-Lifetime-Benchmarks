"""Build manuscript tables from fixed analysis outputs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(output, name, rows):
    path = output / name
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def fmt(value, digits=3):
    return f"{float(value):.{digits}f}"


def median(values):
    values = sorted(float(value) for value in values)
    return values[len(values) // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--interval-analysis", type=Path, required=True)
    parser.add_argument("--metadata-ablation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output directory already exists: {args.output}")
    args.output.mkdir(parents=True)

    flow = read_csv(args.analysis / "cohort_flow.csv")
    excluded = {
        row["cohort_status"]: row
        for row in read_csv(args.analysis / "cohort_exclusion_summary.csv")
    }
    table1 = [
        {"stage": row["stage"].replace("_", " "), "n_cells": row["n_cells"], "detail": ""}
        for row in flow
    ]
    table1[1]["detail"] = "Six candidate cells had no release label because final SOH was at least 0.825."
    table1[3]["detail"] = "Strict >100-cycle rule under both A and B; 503/162/158 train/calibration/test."
    table1.append({
        "stage": "post-release-label exclusions",
        "n_cells": excluded["both_contracts_eol_le_100"]["n_cells"],
        "detail": "All 14 were at or below 100 cycles under both A and B (7 train, 3 calibration, 4 test).",
    })
    outputs = [write_csv(args.output, "table1_cohort_flow.csv", table1)]

    validity = read_csv(args.interval_analysis / "cohort_validity.csv")
    horizon = [row for row in validity if row["stratum_type"] == "overall"]
    outputs.append(write_csv(args.output, "tableS1_dual_contract_horizon.csv", horizon))

    factors = read_csv(args.analysis / "two_by_two_factorization_of_seed_median_mapes.csv")
    table2 = []
    for row in factors:
        table2.append({
            "model": row["model"],
            "AA": fmt(row["mape_AA"]),
            "AB": fmt(row["mape_AB"]),
            "BA": fmt(row["mape_BA"]),
            "BB": fmt(row["mape_BB"]),
            "evaluation_main_effect": fmt(row["evaluation_main_effect"]),
            "fitting_main_effect": fmt(row["fitting_main_effect"]),
            "interaction": fmt(row["interaction"]),
        })
    outputs.append(write_csv(args.output, "table2_point_factorization.csv", table2))

    summary = read_csv(args.analysis / "two_by_two_factor_summary.csv")
    table3 = []
    for model in ("xgboost", "cpmlp", "cptransformer", "ridge"):
        row = {"model": model}
        for metric, key in (
            ("evaluation_main_effect", "evaluation"),
            ("fitting_main_effect", "fitting"),
            ("interaction", "interaction"),
        ):
            source = next(item for item in summary if item["model"] == model and item["metric"] == metric)
            row[key + "_seed_median"] = fmt(source["seed_median"])
            row[key + "_bootstrap_95"] = f"[{fmt(source['bootstrap_95_low'])}, {fmt(source['bootstrap_95_high'])}]"
        table3.append(row)
    outputs.append(write_csv(args.output, "table3_factor_bootstrap.csv", table3))

    branches = read_csv(args.analysis / "endpoint_branch_by_source.csv")
    branch_counts = {}
    for row in branches:
        branch_counts[row["endpoint_kind"]] = branch_counts.get(row["endpoint_kind"], 0) + int(row["n_cells"])
    delta = {
        row["endpoint_kind"]: row
        for row in read_csv(args.analysis / "endpoint_delta_by_branch.csv")
    }
    table4 = []
    for branch in ("observed_first_crossing", "observed_interpolated_crossing", "extrapolated_final20"):
        row = delta[branch]
        table4.append({
            "generator_branch": branch,
            "release_labelled_cells": branch_counts[branch],
            "paired_cells": row["n_cells"],
            "labels_differ": row["n_different"],
            "median_B_minus_A_cycles": fmt(row["median_delta_cycles"], 1),
            "median_absolute_delta_cycles": fmt(row["median_absolute_delta_cycles"], 1),
            "range_B_minus_A_cycles": f"[{fmt(row['minimum_delta_cycles'], 0)}, {fmt(row['maximum_delta_cycles'], 0)}]",
        })
    outputs.append(write_csv(args.output, "tableS2_endpoint_generation_branches.csv", table4))

    robust = []
    for row in read_csv(args.analysis / "robustness_summary.csv"):
        robust.append({
            "condition": row["condition"],
            "seed_42": fmt(row["seed42_B_minus_A"]),
            "seed_2021": fmt(row["seed2021_B_minus_A"]),
            "seed_2024": fmt(row["seed2024_B_minus_A"]),
            "median_B_minus_A": fmt(row["median_B_minus_A"]),
        })
    outputs.append(write_csv(args.output, "table4_robustness_boundaries.csv", robust))

    coverage = []
    for row in read_csv(args.interval_analysis / "source_coverage_matrix.csv"):
        if (
            row["contract"] in {"release", "current_generator"}
            and row["model"] == "cpmlp"
            and row["nominal"] == "0.9"
            and row["method"] == "pooled_absolute"
            and row["is_worst_eligible"] == "True"
        ):
            coverage.append({
                "contract": row["contract"],
                "seed": row["seed"],
                "worst_source": row["dataset_group"],
                "hits": row["hits"],
                "n_test": row["n_finite"],
                "coverage": fmt(row["coverage"]),
                "wilson_95": f"[{fmt(row['coverage_wilson_low'])}, {fmt(row['coverage_wilson_high'])}]",
            })
    outputs.append(write_csv(args.output, "table5_cpmlp_coverage.csv", coverage))

    ablation = read_csv(args.metadata_ablation / "two_by_two_test_metrics.csv")
    table6 = []
    for feature_set in ("early_curve_only", "chemistry_only", "source_and_chemistry"):
        for model in ("xgboost", "ridge"):
            row = {"feature_set": feature_set, "model": model}
            for train, evaluation, key in (
                ("release", "release", "AA"),
                ("release", "current_generator", "AB"),
                ("current_generator", "release", "BA"),
                ("current_generator", "current_generator", "BB"),
            ):
                values = [
                    item["mape"]
                    for item in ablation
                    if item["feature_set"] == feature_set
                    and item["model"] == model
                    and item["train_contract"] == train
                    and item["evaluation_contract"] == evaluation
                ]
                row[key] = fmt(median(values))
            table6.append(row)
    outputs.append(write_csv(args.output, "tableS3_metadata_ablation.csv", table6))

    inputs = (
        args.interval_analysis / "cohort_validity.csv",
        args.interval_analysis / "source_coverage_matrix.csv",
        args.metadata_ablation / "two_by_two_test_metrics.csv",
        args.analysis / "two_by_two_factor_summary.csv",
        args.analysis / "two_by_two_factorization_of_seed_median_mapes.csv",
        args.analysis / "cohort_flow.csv",
        args.analysis / "endpoint_delta_by_branch.csv",
        args.analysis / "robustness_summary.csv",
    )
    manifest = {
        "inputs_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
        "outputs_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in outputs},
        "claim_boundary": "Point-prediction factor effects are conditional on the fixed test set; matched-contract conformal coverage is reported separately.",
    }
    (args.output / "table_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
