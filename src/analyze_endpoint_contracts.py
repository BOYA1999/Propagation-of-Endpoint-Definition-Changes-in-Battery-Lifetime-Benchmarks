"""Compute fixed endpoint-contract contrasts from supplied derived inputs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


CONTRACTS = ("release", "current_generator")
MODELS = ("ridge", "xgboost", "cpmlp", "cptransformer")
ELIGIBLE = ("HUST", "ISU_ILCC", "MATR", "SNL", "Tongji")
METRICS = (
    "mape_AA", "mape_AB", "mape_BA", "mape_BB",
    "evaluation_main_effect", "fitting_main_effect", "interaction",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def point_effects(frame):
    aa = frame["release_release"].mean()
    ab = frame["release_current_generator"].mean()
    ba = frame["current_generator_release"].mean()
    bb = frame["current_generator_current_generator"].mean()
    return {
        "mape_AA": aa,
        "mape_AB": ab,
        "mape_BA": ba,
        "mape_BB": bb,
        "evaluation_main_effect": ((ab - aa) + (bb - ba)) / 2,
        "fitting_main_effect": ((ba - aa) + (bb - ab)) / 2,
        "interaction": bb - ba - ab + aa,
        "evaluation_path_A_fit": ab - aa,
        "fitting_path_B_evaluation": bb - ab,
        "fitting_path_A_evaluation": ba - aa,
        "evaluation_path_B_fit": bb - ba,
    }


def point_error_table(paired, prediction_root):
    labels = {contract: paired.set_index("file_name")[f"{contract}_eol"] for contract in CONTRACTS}
    rows = []
    for fit_contract in CONTRACTS:
        prediction = pd.read_csv(prediction_root / fit_contract / "revealed_predictions.csv")
        prediction = prediction[(prediction["split"] == "test") & prediction["model"].isin(MODELS)].copy()
        for evaluation_contract in CONTRACTS:
            truth = prediction["file_name"].map(labels[evaluation_contract])
            if truth.isna().any():
                raise ValueError("A test prediction lacked a paired evaluation label")
            frame = prediction[["dataset_group", "file_name", "model", "seed", "predicted_eol"]].copy()
            frame["fit_contract"] = fit_contract
            frame["evaluation_contract"] = evaluation_contract
            frame["evaluation_eol"] = truth.to_numpy(float)
            frame["absolute_percentage_error"] = (frame["predicted_eol"] - frame["evaluation_eol"]).abs() / frame["evaluation_eol"]
            rows.append(frame)
    output = pd.concat(rows, ignore_index=True)
    if len(output) != 4 * 3 * 4 * 158:
        raise ValueError(f"Unexpected 2x2 point-error rows: {len(output)}")
    return output


def factor_analysis(errors, reps, seed):
    wide = errors.pivot(index=["model", "seed", "file_name"], columns=["fit_contract", "evaluation_contract"], values="absolute_percentage_error")
    wide.columns = ["_".join(column) for column in wide.columns]
    required = {"release_release", "release_current_generator", "current_generator_release", "current_generator_current_generator"}
    if set(wide.columns) != required or wide.isna().any().any():
        raise ValueError("Incomplete paired 2x2 point-prediction matrix")
    seed_rows, bootstrap_rows, summary_rows = [], [], []
    rng = np.random.default_rng(seed)
    for model in MODELS:
        model_frame = wide.xs(model, level="model").sort_index()
        seeds = sorted(model_frame.index.get_level_values("seed").unique())
        file_orders = []
        seed_frames = []
        for model_seed in seeds:
            frame = model_frame.xs(model_seed, level="seed").sort_index()
            file_orders.append(tuple(frame.index))
            seed_frames.append(frame)
            seed_rows.append({"model": model, "seed": model_seed, **point_effects(frame)})
        if len(seeds) != 3 or len(set(file_orders)) != 1 or len(seed_frames[0]) != 158:
            raise ValueError(f"Test-cell or seed pairing drift for {model}")
        for replicate in range(reps):
            indices = rng.integers(0, len(seed_frames[0]), size=len(seed_frames[0]))
            effects = pd.DataFrame([point_effects(frame.iloc[indices]) for frame in seed_frames])
            bootstrap_rows.append({"model": model, "replicate": replicate, **effects[list(METRICS)].median().to_dict()})
    by_seed = pd.DataFrame(seed_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    for model in MODELS:
        observed = by_seed[by_seed["model"] == model]
        sampled = bootstrap[bootstrap["model"] == model]
        for metric in METRICS:
            summary_rows.append({
                "model": model,
                "metric": metric,
                "seed_median": float(observed[metric].median()),
                "seed_minimum": float(observed[metric].min()),
                "seed_maximum": float(observed[metric].max()),
                "bootstrap_replicates": reps,
                "bootstrap_95_low": float(sampled[metric].quantile(0.025)),
                "bootstrap_95_high": float(sampled[metric].quantile(0.975)),
                "bootstrap_design": "paired nonparametric test-cell bootstrap; the same sampled cells are retained across AA/AB/BA/BB and all three seeds; fitted models and calibration are fixed",
            })
    median_paths = []
    for model in MODELS:
        values = by_seed[by_seed["model"] == model][["mape_AA", "mape_AB", "mape_BA", "mape_BB"]].median()
        aa, ab, ba, bb = (float(values[column]) for column in ("mape_AA", "mape_AB", "mape_BA", "mape_BB"))
        median_paths.append({
            "model": model, "mape_AA": aa, "mape_AB": ab, "mape_BA": ba, "mape_BB": bb,
            "evaluation_main_effect": ((ab - aa) + (bb - ba)) / 2,
            "fitting_main_effect": ((ba - aa) + (bb - ab)) / 2,
            "interaction": bb - ba - ab + aa,
            "definition": "factorization of the four seed-median MAPE cells; descriptive complement to the seed-level and bootstrap analysis",
        })
    return by_seed, bootstrap, pd.DataFrame(summary_rows), pd.DataFrame(median_paths)


def cohort_and_branch_audit(endpoint, paired):
    frame = endpoint.copy()
    frame["released_label_available"] = frame["released_eol"].notna()
    frame["generator_label_available"] = frame["recomputed_eol"].notna()
    frame["release_after_100"] = frame["released_eol"] > 100
    frame["generator_after_100"] = frame["recomputed_eol"] > 100
    frame["included_in_paired_cohort"] = frame["file_name"].isin(set(paired["file_name"]))
    def reason(row):
        if not row.released_label_available:
            return "no_release_label_generator_final_soh_ge_0.825" if row.endpoint_kind == "excluded_final_soh_ge_0.825" else "no_release_label"
        if not row.generator_label_available:
            return "generator_label_not_returned"
        if not row.release_after_100 and not row.generator_after_100:
            return "both_contracts_eol_le_100"
        if not row.release_after_100:
            return "release_eol_le_100_only"
        if not row.generator_after_100:
            return "generator_eol_le_100_only"
        return "included"
    frame["cohort_status"] = frame.apply(reason, axis=1)
    if not (frame["cohort_status"].eq("included") == frame["included_in_paired_cohort"]).all():
        raise ValueError("Cohort classification differs from paired contract membership")
    flow = pd.DataFrame([
        {"stage": "listed_identifiers", "n_cells": len(frame)},
        {"stage": "available_release_labels", "n_cells": int(frame["released_label_available"].sum())},
        {"stage": "available_generator_labels_among_release_labelled", "n_cells": int((frame["released_label_available"] & frame["generator_label_available"]).sum())},
        {"stage": "paired_contracts_strictly_after_cycle_100", "n_cells": int(frame["included_in_paired_cohort"].sum())},
    ])
    exclusions = frame[~frame["included_in_paired_cohort"]].copy()
    exclusion_summary = exclusions.groupby("cohort_status", as_index=False).agg(
        n_cells=("file_name", "size"),
        n_train=("split", lambda values: int((values == "train").sum())),
        n_calibration=("split", lambda values: int((values == "calibration").sum())),
        n_test=("split", lambda values: int((values == "test").sum())),
    )
    branches = frame[frame["released_label_available"]].groupby(["dataset_group", "endpoint_kind"], as_index=False).size().rename(columns={"size": "n_cells"})
    delta = paired.merge(frame[["file_name", "endpoint_kind"]], on="file_name", how="left", validate="one_to_one")
    delta["absolute_label_delta_cycles"] = delta["label_delta_cycles"].abs()
    branch_delta = delta.groupby("endpoint_kind", as_index=False).agg(
        n_cells=("file_name", "size"), n_different=("labels_differ", "sum"),
        median_delta_cycles=("label_delta_cycles", "median"),
        median_absolute_delta_cycles=("absolute_label_delta_cycles", "median"),
        minimum_delta_cycles=("label_delta_cycles", "min"), maximum_delta_cycles=("label_delta_cycles", "max"),
    )
    top5 = delta.sort_values(["absolute_label_delta_cycles", "file_name"], ascending=[False, True]).head(5)
    return frame, flow, exclusions, exclusion_summary, branches, branch_delta, top5


def robustness_summary(gate4):
    rows = []
    for analysis, nominal, label in (
        ("primary_cycle100", 0.9, "cycle-100, 90% nominal"),
        ("primary_cycle100", 0.95, "cycle-100, 95% nominal"),
        ("cycle50", 0.9, "cycle-50, 90% nominal"),
        ("remove_g34c2", 0.9, "cycle-100 without ISU-ILCC_G34C2, 90% nominal"),
    ):
        subset = gate4[(gate4["analysis"] == analysis) & np.isclose(gate4["nominal"].astype(float), nominal)]
        pivot = subset.pivot(index="seed", columns="contract", values="worst_eligible_source_coverage")
        difference = pivot["current_generator"] - pivot["release"]
        rows.append({"condition": label, "analysis": analysis, "nominal": nominal, "seed42_B_minus_A": float(difference.loc[42]), "seed2021_B_minus_A": float(difference.loc[2021]), "seed2024_B_minus_A": float(difference.loc[2024]), "median_B_minus_A": float(difference.median())})
    return pd.DataFrame(rows)


def bootstrap_protocol(paired):
    counts = paired.groupby(["dataset_group", "split"]).size().unstack(fill_value=0)
    eligible = counts.reindex(ELIGIBLE).fillna(0).astype(int)
    if not ((eligible["calibration"] >= 10) & (eligible["test"] >= 10)).all():
        raise ValueError("Source eligibility does not match the fixed cohort counts")
    return {
        "eligibility_rule": "A source is eligible when the fixed 823-cell cohort contains at least 10 calibration cells and at least 10 test cells.",
        "eligible_sources": list(ELIGIBLE),
        "source_counts": {source: {split: int(eligible.loc[source, split]) for split in ("calibration", "test")} for source in ELIGIBLE},
        "target": "difference in the median across three seeds of the minimum source-specific 90% pooled-absolute coverage, current generator minus release",
        "resampling": "Conditional on the observed calibration split: resample test cells with replacement separately within every eligible source, retaining each source test sample size. Use identical within-source draws for both contracts and all three seeds. Calibration residuals and conformal quantiles remain fixed.",
        "worst_source_rule": "Recompute the minimum over all five pre-specified eligible sources in every bootstrap replicate; SNL is not fixed as the worst source during resampling.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-labels", type=Path, required=True)
    parser.add_argument("--endpoint-audit", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--gate4-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paired = pd.read_csv(args.paired_labels)
    endpoint = pd.read_csv(args.endpoint_audit)
    errors = point_error_table(paired, args.prediction_root)
    by_seed, bootstrap, factor_summary, median_paths = factor_analysis(errors, args.bootstrap_reps, args.seed)
    records, flow, exclusions, exclusion_summary, branches, branch_delta, top5 = cohort_and_branch_audit(endpoint, paired)
    robust = robustness_summary(pd.read_csv(args.gate4_metrics))
    outputs = {
        "two_by_two_cellwise_errors.csv": errors,
        "two_by_two_factor_effects_by_seed.csv": by_seed,
        "two_by_two_factor_bootstrap_replicates.csv": bootstrap,
        "two_by_two_factor_summary.csv": factor_summary,
        "two_by_two_factorization_of_seed_median_mapes.csv": median_paths,
        "cohort_flow.csv": flow,
        "cohort_exclusion_records.csv": exclusions,
        "cohort_exclusion_summary.csv": exclusion_summary,
        "endpoint_generation_records.csv": records,
        "endpoint_branch_by_source.csv": branches,
        "endpoint_delta_by_branch.csv": branch_delta,
        "endpoint_top5_label_differences.csv": top5,
        "robustness_summary.csv": robust,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output / name, index=False)
    (args.output / "source_bootstrap_protocol.json").write_text(json.dumps(bootstrap_protocol(paired), indent=2) + "\n", encoding="utf-8")
    manifest = {
        "status": "complete_unvalidated",
        "analysis": "endpoint-contract factorization, branch audit, and boundary checks",
        "inputs_sha256": {"paired_labels": sha256(args.paired_labels), "endpoint_audit": sha256(args.endpoint_audit), "release_predictions": sha256(args.prediction_root / "release" / "revealed_predictions.csv"), "current_generator_predictions": sha256(args.prediction_root / "current_generator" / "revealed_predictions.csv"), "gate4_metrics": sha256(args.gate4_metrics)},
        "parameters": {"bootstrap_reps": args.bootstrap_reps, "seed": args.seed, "models": list(MODELS), "test_cells": 158},
        "source_sha256": sha256(Path(__file__)),
        "outputs_sha256": {path.name: sha256(path) for path in sorted(args.output.iterdir()) if path.is_file()},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
