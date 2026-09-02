import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20260901
BOOTSTRAPS = 5000
MODELS = ("ridge", "xgboost", "cpmlp", "cptransformer")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def order_discordance(a, b):
    i, j = np.triu_indices(len(a), 1)
    sa = np.sign(a[i] - a[j])
    sb = np.sign(b[i] - b[j])
    usable = (sa != 0) & (sb != 0)
    return int((sa[usable] != sb[usable]).sum()), int(usable.sum())


def target_geometry(labels, branches):
    frame = labels.merge(branches[["file_name", "endpoint_kind"]], on="file_name", how="left", validate="one_to_one")
    frame["mean_eol"] = (frame["release_eol"] + frame["current_generator_eol"]) / 2
    frame["absolute_delta_cycles"] = frame["label_delta_cycles"].abs()
    frame["relative_delta"] = frame["label_delta_cycles"] / frame["mean_eol"]
    frame["absolute_relative_delta"] = frame["relative_delta"].abs()
    rows = []
    groups = [("all", "ALL", frame)]
    groups += [("split", key, group) for key, group in frame.groupby("split")]
    groups += [("source", key, group) for key, group in frame.groupby("dataset_group")]
    groups += [("branch", key, group) for key, group in frame.groupby("endpoint_kind")]
    for level, key, group in groups:
        discordant, comparable = order_discordance(group.release_eol.to_numpy(), group.current_generator_eol.to_numpy())
        rows.append({
            "group_level": level,
            "group": key,
            "n_cells": len(group),
            "n_changed": int(group.labels_differ.sum()),
            "median_delta_cycles": float(group.label_delta_cycles.median()),
            "median_absolute_delta_cycles": float(group.absolute_delta_cycles.median()),
            "p90_absolute_delta_cycles": float(group.absolute_delta_cycles.quantile(0.9)),
            "median_absolute_relative_delta": float(group.absolute_relative_delta.median()),
            "spearman_rho": float(spearmanr(group.release_eol, group.current_generator_eol).statistic),
            "kendall_tau_b": float(kendalltau(group.release_eol, group.current_generator_eol).statistic),
            "discordant_pairs": discordant,
            "comparable_pairs": comparable,
            "discordant_pair_fraction": discordant / comparable if comparable else np.nan,
        })
    changed = frame[frame.labels_differ].sort_values("absolute_delta_cycles", ascending=False)
    total = changed.absolute_delta_cycles.sum()
    concentration = []
    for label, count in (("top_1_cell", 1), ("top_5_cells", 5), ("top_10_cells", 10), ("top_10_percent", int(np.ceil(len(changed) * 0.1)))):
        concentration.append({"subset": label, "n_cells": count, "share_of_total_absolute_delta": float(changed.head(count).absolute_delta_cycles.sum() / total)})
    return frame, pd.DataFrame(rows), pd.DataFrame(concentration)


def point_wide(errors, labels):
    frame = errors.copy()
    codes = {("release", "release"): "AA", ("release", "current_generator"): "AB", ("current_generator", "release"): "BA", ("current_generator", "current_generator"): "BB"}
    frame["cell"] = [codes[(a, b)] for a, b in zip(frame.fit_contract, frame.evaluation_contract)]
    wide = frame.pivot(index=["dataset_group", "file_name", "model", "seed"], columns="cell", values="absolute_percentage_error").reset_index()
    predictions = frame[frame.evaluation_contract.eq("release")].pivot(index=["dataset_group", "file_name", "model", "seed"], columns="fit_contract", values="predicted_eol").reset_index()
    predictions = predictions.rename(columns={"release": "prediction_A", "current_generator": "prediction_B"})
    wide = wide.merge(predictions, on=["dataset_group", "file_name", "model", "seed"], validate="one_to_one")
    wide = wide.merge(labels[["file_name", "release_eol", "current_generator_eol", "label_delta_cycles", "labels_differ"]], on="file_name", validate="many_to_one")
    wide["evaluation_effect"] = ((wide.AB - wide.AA) + (wide.BB - wide.BA)) / 2
    wide["fitting_effect"] = ((wide.BA - wide.AA) + (wide.BB - wide.AB)) / 2
    wide["interaction"] = wide.BB - wide.BA - wide.AB + wide.AA
    wide["prediction_shift"] = wide.prediction_B - wide.prediction_A
    wide["absolute_prediction_shift"] = wide.prediction_shift.abs()
    return wide


def propagation_analysis(wide):
    subgroup_rows = []
    for (model, seed, changed), group in wide.groupby(["model", "seed", "labels_differ"]):
        n_total = int(((wide.model == model) & (wide.seed == seed)).sum())
        subgroup_rows.append({
            "model": model, "seed": seed, "label_status": "changed" if changed else "unchanged", "n_cells": len(group),
            "evaluation_effect_within_group": float(group.evaluation_effect.mean()),
            "fitting_effect_within_group": float(group.fitting_effect.mean()),
            "interaction_within_group": float(group.interaction.mean()),
            "evaluation_contribution_to_full_mean": float(group.evaluation_effect.sum() / n_total),
            "fitting_contribution_to_full_mean": float(group.fitting_effect.sum() / n_total),
            "interaction_contribution_to_full_mean": float(group.interaction.sum() / n_total),
            "median_absolute_prediction_shift": float(group.absolute_prediction_shift.median()),
        })
    source_rows, loo_rows, shift_rows = [], [], []
    for (model, seed), group in wide.groupby(["model", "seed"]):
        n_total = len(group)
        changed = group[group.labels_differ]
        total_shift = group.absolute_prediction_shift.sum()
        shift_rows.append({
            "model": model, "seed": seed,
            "unchanged_cell_share_of_absolute_prediction_shift": float(group.loc[~group.labels_differ, "absolute_prediction_shift"].sum() / total_shift) if total_shift else 0.0,
            "median_absolute_shift_changed": float(changed.absolute_prediction_shift.median()),
            "median_absolute_shift_unchanged": float(group.loc[~group.labels_differ, "absolute_prediction_shift"].median()),
            "prediction_label_shift_spearman_changed": float(spearmanr(changed.prediction_shift, changed.label_delta_cycles).statistic),
        })
        for source, source_group in group.groupby("dataset_group"):
            source_rows.append({
                "model": model, "seed": seed, "source": source, "n_cells": len(source_group),
                "evaluation_contribution_to_full_mean": float(source_group.evaluation_effect.sum() / n_total),
                "fitting_contribution_to_full_mean": float(source_group.fitting_effect.sum() / n_total),
                "interaction_contribution_to_full_mean": float(source_group.interaction.sum() / n_total),
            })
            kept = group[group.dataset_group.ne(source)]
            loo_rows.append({"model": model, "seed": seed, "excluded_source": source, "n_cells": len(kept), "evaluation_effect": float(kept.evaluation_effect.mean()), "fitting_effect": float(kept.fitting_effect.mean()), "interaction": float(kept.interaction.mean())})
    rng = np.random.default_rng(SEED)
    boot_rows = []
    for model, group in wide.groupby("model"):
        files = sorted(group.file_name.unique())
        seed_frames = [group[group.seed.eq(seed)].set_index("file_name").loc[files] for seed in sorted(group.seed.unique())]
        for replicate in range(BOOTSTRAPS):
            idx = rng.integers(0, len(files), len(files))
            values = []
            for frame in seed_frames:
                sampled = frame.iloc[idx]
                total_shift = sampled.absolute_prediction_shift.sum()
                values.append({
                    "evaluation_effect": sampled.evaluation_effect.mean(),
                    "fitting_effect": sampled.fitting_effect.mean(),
                    "unchanged_shift_share": sampled.loc[~sampled.labels_differ, "absolute_prediction_shift"].sum() / total_shift if total_shift else 0.0,
                })
            med = pd.DataFrame(values).median()
            boot_rows.append({"model": model, "replicate": replicate, **med.to_dict()})
    bootstrap = pd.DataFrame(boot_rows)
    summary = []
    for model, group in bootstrap.groupby("model"):
        for metric in ("evaluation_effect", "fitting_effect", "unchanged_shift_share"):
            summary.append({"model": model, "metric": metric, "bootstrap_median": float(group[metric].median()), "bootstrap_95_low": float(group[metric].quantile(0.025)), "bootstrap_95_high": float(group[metric].quantile(0.975)), "replicates": BOOTSTRAPS})
    return pd.DataFrame(subgroup_rows), pd.DataFrame(source_rows), pd.DataFrame(loo_rows), pd.DataFrame(shift_rows), pd.DataFrame(summary)


def metric_analysis(errors):
    frame = errors.copy()
    residual = (frame.predicted_eol - frame.evaluation_eol).abs()
    frame["MAPE"] = residual / frame.evaluation_eol
    frame["MAE_cycles"] = residual
    frame["RMSE_cycles"] = residual ** 2
    frame["sMAPE"] = 2 * residual / (frame.predicted_eol.abs() + frame.evaluation_eol.abs())
    frame["absolute_log_error"] = (np.log(frame.predicted_eol) - np.log(frame.evaluation_eol)).abs()
    codes = {("release", "release"): "AA", ("release", "current_generator"): "AB", ("current_generator", "release"): "BA", ("current_generator", "current_generator"): "BB"}
    frame["cell"] = [codes[(a, b)] for a, b in zip(frame.fit_contract, frame.evaluation_contract)]
    rows = []
    for (model, seed), group in frame.groupby(["model", "seed"]):
        for metric in ("MAPE", "MAE_cycles", "RMSE_cycles", "sMAPE", "absolute_log_error"):
            cells = group.groupby("cell")[metric].mean()
            if metric == "RMSE_cycles":
                cells = np.sqrt(cells)
            aa, ab, ba, bb = (float(cells[key]) for key in ("AA", "AB", "BA", "BB"))
            rows.append({"model": model, "seed": seed, "metric": metric, "AA": aa, "AB": ab, "BA": ba, "BB": bb, "evaluation_effect": ((ab-aa)+(bb-ba))/2, "fitting_effect": ((ba-aa)+(bb-ab))/2, "interaction": bb-ba-ab+aa})
    by_seed = pd.DataFrame(rows)
    summary = by_seed.groupby(["model", "metric"], as_index=False).agg(evaluation_median=("evaluation_effect", "median"), evaluation_min=("evaluation_effect", "min"), evaluation_max=("evaluation_effect", "max"), fitting_median=("fitting_effect", "median"), fitting_min=("fitting_effect", "min"), fitting_max=("fitting_effect", "max"), interaction_median=("interaction", "median"))
    return by_seed, summary


def interval_analysis(intervals, gate4, labels):
    frame = intervals[(intervals.model.eq("cpmlp")) & np.isclose(intervals.nominal, 0.9) & intervals.method.eq("pooled_absolute")].copy()
    keys = ["seed", "file_name", "dataset_group"]
    a = frame[frame.contract.eq("release")][keys + ["eol_cycle", "predicted_eol", "calibration_quantile", "covered"]].rename(columns={"eol_cycle": "y_A", "predicted_eol": "p_A", "calibration_quantile": "q_A", "covered": "covered_A"})
    b = frame[frame.contract.eq("current_generator")][keys + ["eol_cycle", "predicted_eol", "calibration_quantile", "covered"]].rename(columns={"eol_cycle": "y_B", "predicted_eol": "p_B", "calibration_quantile": "q_B", "covered": "covered_B"})
    merged = a.merge(b, on=keys, validate="one_to_one")
    g = lambda y, p: -np.abs(y-p)
    merged["margin_A"] = merged.q_A + g(merged.y_A, merged.p_A)
    merged["margin_B"] = merged.q_B + g(merged.y_B, merged.p_B)
    merged["delta_margin"] = merged.margin_B - merged.margin_A
    merged["target_component"] = 0.5 * ((g(merged.y_B, merged.p_A)-g(merged.y_A, merged.p_A)) + (g(merged.y_B, merged.p_B)-g(merged.y_A, merged.p_B)))
    merged["centre_component"] = 0.5 * ((g(merged.y_A, merged.p_B)-g(merged.y_A, merged.p_A)) + (g(merged.y_B, merged.p_B)-g(merged.y_B, merged.p_A)))
    merged["radius_component"] = merged.q_B - merged.q_A
    merged["transition"] = np.select([merged.covered_A & merged.covered_B, merged.covered_A & ~merged.covered_B, ~merged.covered_A & merged.covered_B], ["both", "A_only", "B_only"], default="neither")
    transition = merged.groupby(keys[:1] + ["dataset_group", "transition"], as_index=False).size().rename(columns={"size": "n_cells"})
    coverage = merged.groupby(["seed", "dataset_group"], as_index=False).agg(n_test=("file_name", "size"), covered_A=("covered_A", "sum"), covered_B=("covered_B", "sum"), mean_target_component=("target_component", "mean"), mean_centre_component=("centre_component", "mean"), mean_radius_component=("radius_component", "mean"), mean_delta_margin=("delta_margin", "mean"))
    coverage["coverage_A"] = coverage.covered_A / coverage.n_test
    coverage["coverage_B"] = coverage.covered_B / coverage.n_test
    coverage["coverage_B_minus_A"] = coverage.coverage_B - coverage.coverage_A
    coverage["coverage_step"] = 1 / coverage.n_test
    primary = gate4[(gate4.analysis.eq("primary_cycle100")) & np.isclose(gate4.nominal, 0.9)].copy()
    removal = gate4[(gate4.analysis.eq("remove_g34c2")) & np.isclose(gate4.nominal, 0.9)].copy()
    compare_keys = ["contract", "seed", "file_name", "dataset_group"]
    primary = primary[compare_keys + ["predicted_eol", "calibration_quantile", "covered"]].rename(columns={"predicted_eol": "prediction_primary", "calibration_quantile": "q_primary", "covered": "covered_primary"})
    removal = removal[compare_keys + ["predicted_eol", "calibration_quantile", "covered"]].rename(columns={"predicted_eol": "prediction_removed", "calibration_quantile": "q_removed", "covered": "covered_removed"})
    leverage = primary.merge(removal, on=compare_keys, validate="one_to_one")
    leverage["prediction_shift_after_removal"] = leverage.prediction_removed - leverage.prediction_primary
    leverage["quantile_shift_after_removal"] = leverage.q_removed - leverage.q_primary
    leverage["coverage_transition_after_removal"] = np.select([leverage.covered_primary & leverage.covered_removed, leverage.covered_primary & ~leverage.covered_removed, ~leverage.covered_primary & leverage.covered_removed], ["both", "primary_only", "removed_only"], default="neither")
    leverage_summary = leverage.groupby(["contract", "seed", "dataset_group"], as_index=False).agg(n_cells=("file_name", "size"), median_prediction_shift=("prediction_shift_after_removal", "median"), median_absolute_prediction_shift=("prediction_shift_after_removal", lambda x: float(np.median(np.abs(x)))), quantile_shift=("quantile_shift_after_removal", "first"), covered_primary=("covered_primary", "sum"), covered_removed=("covered_removed", "sum"))
    leverage_summary["coverage_change"] = (leverage_summary.covered_removed - leverage_summary.covered_primary) / leverage_summary.n_cells
    g34 = labels[labels.file_name.eq("ISU-ILCC_G34C2.pkl")].copy()
    return merged, transition, coverage, leverage, leverage_summary, g34


def identifiability(features, labels):
    frame = features.merge(labels[["file_name", "labels_differ"]], on="file_name", validate="one_to_one")
    frame = frame[frame.dataset_group.eq("ISU_ILCC")].copy()
    feature_columns = [column for column in features.columns if column not in {"split", "dataset_group", "chemistry_system", "file_name"}]
    train = frame[frame.split.eq("train")]
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LogisticRegression(C=1.0, l1_ratio=0, class_weight="balanced", solver="liblinear", random_state=SEED, max_iter=5000))])
    pipe.fit(train[feature_columns], train.labels_differ.astype(int))
    rows, predictions = [], []
    for split in ("calibration", "test"):
        part = frame[frame.split.eq(split)].copy()
        y = part.labels_differ.astype(int).to_numpy()
        score = pipe.predict_proba(part[feature_columns])[:, 1]
        rows.append({"split": split, "n_cells": len(part), "n_changed": int(y.sum()), "prevalence": float(y.mean()), "auroc": float(roc_auc_score(y, score)), "average_precision": float(average_precision_score(y, score)), "brier_score": float(brier_score_loss(y, score))})
        predictions.extend({"split": split, "file_name": name, "labels_differ": int(label), "predicted_probability": float(prob)} for name, label, prob in zip(part.file_name, y, score))
    pred = pd.DataFrame(predictions)
    test = pred[pred.split.eq("test")]
    rng = np.random.default_rng(SEED)
    pos = test.index[test.labels_differ.eq(1)].to_numpy()
    neg = test.index[test.labels_differ.eq(0)].to_numpy()
    boot = []
    for _ in range(BOOTSTRAPS):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])
        sample = test.loc[idx]
        boot.append((roc_auc_score(sample.labels_differ, sample.predicted_probability), average_precision_score(sample.labels_differ, sample.predicted_probability)))
    boot = np.asarray(boot)
    summary = pd.DataFrame(rows)
    summary.loc[summary.split.eq("test"), "auroc_bootstrap_95_low"] = np.quantile(boot[:, 0], 0.025)
    summary.loc[summary.split.eq("test"), "auroc_bootstrap_95_high"] = np.quantile(boot[:, 0], 0.975)
    summary.loc[summary.split.eq("test"), "average_precision_bootstrap_95_low"] = np.quantile(boot[:, 1], 0.025)
    summary.loc[summary.split.eq("test"), "average_precision_bootstrap_95_high"] = np.quantile(boot[:, 1], 0.975)
    return summary, pred, feature_columns


def main():
    global SEED, BOOTSTRAPS
    parser = argparse.ArgumentParser(description="Run the endpoint geometry and propagation analyses.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--branches", type=Path, required=True)
    parser.add_argument("--errors", type=Path, required=True)
    parser.add_argument("--intervals", type=Path, required=True)
    parser.add_argument("--gate4", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAPS)
    args = parser.parse_args()
    SEED = args.seed
    BOOTSTRAPS = args.bootstrap_reps
    output = args.output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    inputs = {
        "labels": args.labels,
        "branches": args.branches,
        "errors": args.errors,
        "intervals": args.intervals,
        "gate4": args.gate4,
        "features": args.features,
    }
    data = {key: pd.read_csv(path) for key, path in inputs.items()}
    geometry_cells, geometry_summary, concentration = target_geometry(data["labels"], data["branches"])
    wide = point_wide(data["errors"], data["labels"])
    propagation = propagation_analysis(wide)
    metric_by_seed, metric_summary = metric_analysis(data["errors"])
    interval = interval_analysis(data["intervals"], data["gate4"], data["labels"])
    identity_summary, identity_predictions, identity_features = identifiability(data["features"], data["labels"])
    outputs = {
        "target_geometry_cells.csv": geometry_cells,
        "target_geometry_summary.csv": geometry_summary,
        "target_delta_concentration.csv": concentration,
        "cellwise_propagation.csv": wide,
        "propagation_by_label_status.csv": propagation[0],
        "source_effect_contributions.csv": propagation[1],
        "leave_one_source_out_effects.csv": propagation[2],
        "prediction_shift_summary.csv": propagation[3],
        "propagation_bootstrap_summary.csv": propagation[4],
        "metric_factor_effects_by_seed.csv": metric_by_seed,
        "metric_factor_summary.csv": metric_summary,
        "interval_margin_decomposition_cells.csv": interval[0],
        "interval_coverage_transitions.csv": interval[1],
        "interval_margin_summary_by_source.csv": interval[2],
        "g34c2_training_leverage_cells.csv": interval[3],
        "g34c2_training_leverage_summary.csv": interval[4],
        "g34c2_record.csv": interval[5],
        "isu_early_identifiability_summary.csv": identity_summary,
        "isu_early_identifiability_predictions.csv": identity_predictions,
    }
    for name, frame in outputs.items():
        frame.to_csv(output / name, index=False)
    all_geometry = geometry_summary[(geometry_summary.group_level.eq("all")) & geometry_summary.group.eq("ALL")].iloc[0]
    snl = interval[2][interval[2].dataset_group.eq("SNL")]
    test_identity = identity_summary[identity_summary.split.eq("test")].iloc[0]
    summary = {
        "status": "complete_unvalidated",
        "headline": {
            "changed_cells": int(data["labels"].labels_differ.sum()),
            "global_spearman_rho": float(all_geometry.spearman_rho),
            "global_discordant_pair_fraction": float(all_geometry.discordant_pair_fraction),
            "snl_coverage_differences": {str(int(row.seed)): float(row.coverage_B_minus_A) for _, row in snl.iterrows()},
            "isu_test_early_feature_auroc": float(test_identity.auroc),
            "isu_test_early_feature_auroc_95": [float(test_identity.auroc_bootstrap_95_low), float(test_identity.auroc_bootstrap_95_high)],
        },
        "interpretation_boundary": "post hoc mechanism analysis; fixed data, split, models and predictions; no endpoint ground truth and no external validation",
    }
    (output / "campaign_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete_unvalidated",
        "campaign": "v10 endpoint geometry and propagation mechanisms",
        "parameters": {"seed": SEED, "bootstrap_replicates": BOOTSTRAPS, "models": list(MODELS), "identifiability_features": identity_features},
        "inputs_sha256": {key: sha256(path) for key, path in inputs.items()},
        "source_sha256": sha256(Path(__file__)),
        "outputs_sha256": {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
