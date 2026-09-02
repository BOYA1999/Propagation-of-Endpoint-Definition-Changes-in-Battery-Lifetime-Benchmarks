"""Validate the derived endpoint-contract analysis without recomputing it."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.core / "manifest.json").read_text(encoding="utf-8"))
    checks = {"manifest_hashes": all(sha256(args.core / name) == digest for name, digest in manifest["outputs_sha256"].items())}
    errors = pd.read_csv(args.core / "two_by_two_cellwise_errors.csv")
    checks["paired_cellwise_2x2_complete"] = len(errors) == 7584 and errors.groupby(["model", "seed", "fit_contract", "evaluation_contract"])["file_name"].nunique().eq(158).all()
    effects = pd.read_csv(args.core / "two_by_two_factor_effects_by_seed.csv")
    summary = pd.read_csv(args.core / "two_by_two_factor_summary.csv")
    bootstrap = pd.read_csv(args.core / "two_by_two_factor_bootstrap_replicates.csv")
    medians = pd.read_csv(args.core / "two_by_two_factorization_of_seed_median_mapes.csv")
    checks["seed_factor_effects_complete"] = len(effects) == 12 and len(summary) == 28 and np.isfinite(effects.select_dtypes("number")).all().all()
    checks["paired_bootstrap_complete"] = len(bootstrap) == 40000 and bootstrap.groupby("model")["replicate"].nunique().eq(10000).all() and np.isfinite(bootstrap.select_dtypes("number")).all().all()
    checks["median_path_factorization_correct"] = bool(np.allclose(medians["evaluation_main_effect"], ((medians["mape_AB"] - medians["mape_AA"]) + (medians["mape_BB"] - medians["mape_BA"])) / 2) and np.allclose(medians["fitting_main_effect"], ((medians["mape_BA"] - medians["mape_AA"]) + (medians["mape_BB"] - medians["mape_AB"])) / 2) and np.allclose(medians["interaction"], medians["mape_BB"] - medians["mape_BA"] - medians["mape_AB"] + medians["mape_AA"]))
    flow = pd.read_csv(args.core / "cohort_flow.csv").set_index("stage")["n_cells"]
    exclusions = pd.read_csv(args.core / "cohort_exclusion_summary.csv").set_index("cohort_status")["n_cells"]
    checks["cohort_flow_and_exclusions"] = flow.to_dict() == {"listed_identifiers": 843, "available_release_labels": 837, "available_generator_labels_among_release_labelled": 837, "paired_contracts_strictly_after_cycle_100": 823} and exclusions.to_dict() == {"both_contracts_eol_le_100": 14, "no_release_label_generator_final_soh_ge_0.825": 6}
    branches = pd.read_csv(args.core / "endpoint_branch_by_source.csv")
    checks["endpoint_branches_complete"] = len(branches) > 0 and branches["n_cells"].sum() == 837 and set(branches["endpoint_kind"]) == {"observed_first_crossing", "observed_interpolated_crossing", "extrapolated_final20"}
    top5 = pd.read_csv(args.core / "endpoint_top5_label_differences.csv")
    checks["top_difference_branch_traceable"] = len(top5) == 5 and top5.iloc[0]["file_name"] == "ISU-ILCC_G34C2.pkl" and top5["endpoint_kind"].notna().all()
    robust = pd.read_csv(args.core / "robustness_summary.csv").set_index(["analysis", "nominal"])["median_B_minus_A"]
    checks["robustness_boundaries_retained"] = np.isclose(robust.loc[("primary_cycle100", 0.95)], 0.0) and np.isclose(robust.loc[("cycle50", 0.9)], 0.0) and np.isclose(robust.loc[("remove_g34c2", 0.9)], -0.1)
    protocol = json.loads((args.core / "source_bootstrap_protocol.json").read_text(encoding="utf-8"))
    checks["source_bootstrap_protocol_complete"] = protocol["eligible_sources"] == ["HUST", "ISU_ILCC", "MATR", "SNL", "Tongji"] and all(values["calibration"] >= 10 and values["test"] >= 10 for values in protocol["source_counts"].values()) and "conditional" in protocol["resampling"].lower() and "recompute" in protocol["worst_source_rule"].lower()
    checks = {name: bool(value) for name, value in checks.items()}
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "passed": sum(checks.values()), "total": len(checks)}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
