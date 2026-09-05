import argparse
import hashlib
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = tuple(range(20)) + (42, 2021, 2024)
CONTRACTS = ("release", "current_generator")
MODELS = ("ridge", "xgboost", "cpmlp", "cptransformer")
METRICS = ("mape", "smape", "mae", "rmse", "log_mae")
NOMINALS = (0.9, 0.95)
BOOTSTRAP_SEED = 130904
SPLIT_COUNTS = {"train": 503, "calibration": 162, "test": 158}
PAIRING = "same seed, initialization, row order, batch order, features, architecture, optimizer and epoch budget; training labels only differ"
XGBOOST = {
    "n_estimators": 700,
    "max_depth": 4,
    "learning_rate": 0.03,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
}
ANALYSIS_OUTPUTS = (
    "revealed_test_predictions.csv",
    "point_metrics_by_seed.csv",
    "factor_contrasts_by_seed.csv",
    "model_ranks_by_cell.csv",
    "ranking_comparisons.csv",
    "best_model_frequency.csv",
    "pairwise_model_wins.csv",
    "prediction_shift_cells.csv",
    "cross_contract_prediction_shift_by_seed.csv",
    "within_contract_refit_pairwise.csv",
    "refit_variability_summary.csv",
    "factor_seed_cell_bootstrap_replicates.csv",
    "factor_seed_cell_bootstrap_summary.csv",
    "prediction_shift_bootstrap_replicates.csv",
    "prediction_shift_bootstrap_summary.csv",
    "conformal_metrics.csv",
    "calibration_quantile_neighbourhood.csv",
    "conformal_paired_differences_by_seed.csv",
    "conformal_paired_summary.csv",
    "xgboost_pipeline_vs_controlled.csv",
    "training_run_summary.csv",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric(y, prediction, name):
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = np.abs(y - prediction)
    if name == "mape":
        return float(np.mean(error / y))
    if name == "smape":
        return float(np.mean(2 * error / (np.abs(y) + np.abs(prediction))))
    if name == "mae":
        return float(np.mean(error))
    if name == "rmse":
        return float(np.sqrt(np.mean((y - prediction) ** 2)))
    return float(np.mean(np.abs(np.log(y) - np.log(prediction))))


def contribution(y, prediction, name):
    error = np.abs(y - prediction)
    if name == "mape":
        return error / y
    if name == "smape":
        return 2 * error / (np.abs(y) + np.abs(prediction))
    if name == "mae":
        return error
    if name == "rmse":
        return (y - prediction) ** 2
    return np.abs(np.log(y) - np.log(prediction))


def finite_quantile(values, nominal):
    ordered = np.sort(np.asarray(values, dtype=float))
    k = int(np.ceil((len(ordered) + 1) * nominal))
    return (float(ordered[k - 1]) if k <= len(ordered) else np.nan), k


def interval_score(y, lower, upper, nominal):
    alpha = 1 - nominal
    return upper - lower + np.where(y < lower, 2 * (lower - y) / alpha, 0) + np.where(y > upper, 2 * (y - upper) / alpha, 0)


def add_check(report, name, passed, **details):
    report["checks"][name] = {"passed": bool(passed), **details}


def normalized_text(series):
    return series.where(series.notna(), "").astype(str).str.lower().replace({"true": "true", "false": "false"})


def compare_frame(report, name, expected, path, keys):
    if not path.exists():
        add_check(report, name, False, error=f"missing: {path}")
        return False
    actual = pd.read_csv(path)
    missing = sorted(set(expected.columns) - set(actual.columns))
    extra = sorted(set(actual.columns) - set(expected.columns))
    if missing or extra:
        add_check(report, name, False, expected_rows=len(expected), actual_rows=len(actual), missing_columns=missing, extra_columns=extra)
        return False
    if actual.duplicated(keys).any() or expected.duplicated(keys).any():
        add_check(report, name, False, expected_rows=len(expected), actual_rows=len(actual), duplicate_key_rows=int(actual.duplicated(keys, keep=False).sum()))
        return False
    expected = expected.sort_values(keys).reset_index(drop=True)
    actual = actual.sort_values(keys).reset_index(drop=True)
    if len(expected) != len(actual):
        add_check(report, name, False, expected_rows=len(expected), actual_rows=len(actual))
        return False
    failures = []
    max_error = 0.0
    for column in expected.columns:
        if pd.api.types.is_numeric_dtype(expected[column]):
            left = pd.to_numeric(expected[column], errors="coerce").to_numpy(float)
            right = pd.to_numeric(actual[column], errors="coerce").to_numpy(float)
            finite = np.isfinite(left) & np.isfinite(right)
            if finite.any():
                max_error = max(max_error, float(np.max(np.abs(left[finite] - right[finite]))))
            if not np.allclose(left, right, rtol=1e-10, atol=1e-10, equal_nan=True):
                failures.append(column)
        elif not normalized_text(expected[column]).equals(normalized_text(actual[column])):
            failures.append(column)
    passed = not failures
    add_check(report, name, passed, expected_rows=len(expected), actual_rows=len(actual), max_absolute_error=max_error, mismatched_columns=failures)
    return passed


def resolve(project_root, value):
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_labels(root):
    labels = {}
    for contract in CONTRACTS:
        labels[contract] = {}
        for split, file_name in (("train", "train_labels.csv"), ("calibration", "calibration_labels.csv"), ("test", "test_labels_sealed.csv")):
            frame = pd.read_csv(root / contract / file_name)
            if frame.file_name.duplicated().any() or not np.isfinite(frame.eol_cycle).all():
                raise ValueError(f"invalid labels: {contract}/{split}")
            labels[contract][split] = frame.set_index("file_name").eol_cycle.astype(float)
    return labels


def point_tables(predictions, labels):
    revealed = []
    for evaluation_contract in CONTRACTS:
        frame = predictions[predictions.split.eq("test")].copy()
        frame = frame.merge(labels[evaluation_contract]["test"].rename("eol_cycle"), left_on="file_name", right_index=True, how="left", validate="many_to_one")
        frame["evaluation_contract"] = evaluation_contract
        revealed.append(frame)
    revealed = pd.concat(revealed, ignore_index=True)
    point_rows = []
    for keys, group in revealed.groupby(["fit_contract", "evaluation_contract", "model", "seed"], sort=True):
        row = {"fit_contract": keys[0], "evaluation_contract": keys[1], "model": keys[2], "seed": int(keys[3]), "n_test": len(group)}
        row.update({name: metric(group.eol_cycle, group.predicted_eol, name) for name in METRICS})
        point_rows.append(row)
    points = pd.DataFrame(point_rows)
    factor_rows = []
    for (model, seed), group in points.groupby(["model", "seed"], sort=True):
        cells = {(row.fit_contract, row.evaluation_contract): row for row in group.itertuples()}
        for name in METRICS:
            values = {
                "AA": getattr(cells[("release", "release")], name),
                "AB": getattr(cells[("release", "current_generator")], name),
                "BA": getattr(cells[("current_generator", "release")], name),
                "BB": getattr(cells[("current_generator", "current_generator")], name),
            }
            factor_rows.append({
                "model": model,
                "seed": int(seed),
                "metric": name,
                **values,
                "evaluation_contrast": ((values["AB"] - values["AA"]) + (values["BB"] - values["BA"])) / 2,
                "fitting_contrast": ((values["BA"] - values["AA"]) + (values["BB"] - values["AB"])) / 2,
                "interaction": values["BB"] - values["BA"] - values["AB"] + values["AA"],
            })
    return points, pd.DataFrame(factor_rows)


def rank_table(points):
    cell = {
        ("release", "release"): "AA",
        ("release", "current_generator"): "AB",
        ("current_generator", "release"): "BA",
        ("current_generator", "current_generator"): "BB",
    }
    rows = []
    for row in points.itertuples():
        for name in METRICS:
            rows.append({"cell": cell[(row.fit_contract, row.evaluation_contract)], "model": row.model, "seed": int(row.seed), "metric": name, "score": getattr(row, name)})
    ranks = pd.DataFrame(rows)
    ranks["rank"] = ranks.groupby(["cell", "seed", "metric"])["score"].rank(method="min")
    return ranks


def best_frequency(ranks):
    matched = ranks[ranks.cell.isin(["AA", "BB"])]
    counts = matched[matched["rank"].eq(1)].groupby(["cell", "metric", "model"], as_index=False).size().rename(columns={"size": "best_seed_count"})
    grid = pd.MultiIndex.from_product([["AA", "BB"], METRICS, MODELS], names=["cell", "metric", "model"]).to_frame(index=False)
    best = grid.merge(counts, on=["cell", "metric", "model"], how="left")
    best["best_seed_count"] = best.best_seed_count.fillna(0).astype(int)
    best["total_seeds"] = matched.seed.nunique()
    best["best_seed_fraction"] = best.best_seed_count / best.total_seeds
    best["tie_handling"] = "exact rank-1 ties count as co-winners"
    return best


def parse_paired(path):
    paired = pd.read_csv(path)
    paired["labels_differ"] = paired.labels_differ.astype(str).str.lower().eq("true")
    if paired.file_name.duplicated().any():
        raise ValueError("duplicate paired-label identifiers")
    return paired


def variability_tables(predictions, paired):
    test = predictions[predictions.split.eq("test")].merge(paired[["file_name", "labels_differ", "label_delta_cycles"]], on="file_name", validate="many_to_one")
    wide = test.pivot(index=["dataset_group", "file_name", "model", "seed", "labels_differ", "label_delta_cycles"], columns="fit_contract", values="predicted_eol").reset_index()
    wide["signed_shift_B_minus_A"] = wide.current_generator - wide.release
    wide["absolute_shift"] = wide.signed_shift_B_minus_A.abs()
    cross_rows = []
    for (model, seed), group in wide.groupby(["model", "seed"], sort=True):
        for status, subset in (("all", group), ("changed", group[group.labels_differ]), ("unchanged", group[~group.labels_differ])):
            cross_rows.append({
                "model": model,
                "seed": int(seed),
                "label_status": status,
                "n_cells": len(subset),
                "mean_absolute_shift": float(subset.absolute_shift.mean()),
                "median_absolute_shift": float(subset.absolute_shift.median()),
                "mean_signed_shift_B_minus_A": float(subset.signed_shift_B_minus_A.mean()),
            })
    cross = pd.DataFrame(cross_rows)
    within_rows = []
    for model in MODELS:
        current = wide[wide.model.eq(model)].sort_values(["seed", "file_name"])
        matrices = {contract: current.pivot(index="seed", columns="file_name", values=contract) for contract in CONTRACTS}
        metadata = current.drop_duplicates("file_name").set_index("file_name").loc[matrices["release"].columns]
        for contract, matrix in matrices.items():
            for seed_left, seed_right in itertools.combinations(matrix.index, 2):
                shift = np.abs(matrix.loc[seed_left].to_numpy() - matrix.loc[seed_right].to_numpy())
                for status, mask in (("all", np.ones(len(metadata), dtype=bool)), ("changed", metadata.labels_differ.to_numpy(bool)), ("unchanged", ~metadata.labels_differ.to_numpy(bool))):
                    within_rows.append({
                        "model": model,
                        "fit_contract": contract,
                        "seed_left": int(seed_left),
                        "seed_right": int(seed_right),
                        "label_status": status,
                        "n_cells": int(mask.sum()),
                        "mean_absolute_shift": float(np.mean(shift[mask])),
                        "median_absolute_shift": float(np.median(shift[mask])),
                    })
    within = pd.DataFrame(within_rows)
    summaries = []
    for model in MODELS:
        for status in ("all", "changed", "unchanged"):
            cross_values = cross[(cross.model.eq(model)) & (cross.label_status.eq(status))].mean_absolute_shift
            release = within[(within.model.eq(model)) & (within.fit_contract.eq("release")) & (within.label_status.eq(status))].mean_absolute_shift
            current = within[(within.model.eq(model)) & (within.fit_contract.eq("current_generator")) & (within.label_status.eq(status))].mean_absolute_shift
            pooled = pd.concat([release, current], ignore_index=True)
            baseline = float(pooled.median())
            summaries.append({
                "model": model,
                "label_status": status,
                "cross_contract_seed_median_mean_absolute_shift": float(cross_values.median()),
                "within_release_pair_median_mean_absolute_shift": float(release.median()),
                "within_current_pair_median_mean_absolute_shift": float(current.median()),
                "within_pooled_pair_median_mean_absolute_shift": baseline,
                "cross_to_within_ratio": float(cross_values.median() / baseline) if baseline > 0 else np.nan,
                "cross_seed_fraction_above_within_median": float((cross_values > baseline).mean()),
                "cross_seed_count": len(cross_values),
                "within_pair_count_per_contract": len(release),
            })
    return wide, cross, within, pd.DataFrame(summaries)


def source_draws(sources, reps, rng):
    sources = np.asarray(sources)
    return np.concatenate([rng.choice(np.flatnonzero(sources == source), size=(reps, int((sources == source).sum())), replace=True) for source in sorted(set(sources))], axis=1)


def boot_seed_metrics(values, seed_draws, cell_draws, rmse=False):
    output = np.empty(seed_draws.shape, dtype=float)
    for start in range(0, len(seed_draws), 100):
        stop = min(start + 100, len(seed_draws))
        selected = values[seed_draws[start:stop, :, None], cell_draws[start:stop, None, :]]
        per_seed = selected.mean(axis=2)
        output[start:stop] = np.sqrt(per_seed) if rmse else per_seed
    return output


def factor_bootstrap(predictions, labels, reps):
    test = predictions[predictions.split.eq("test")]
    seeds = np.sort(test.seed.unique())
    files = test[test.model.eq(MODELS[0])].drop_duplicates("file_name").sort_values("file_name")
    file_order = files.file_name.tolist()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    seed_draws = rng.integers(0, len(seeds), size=(reps, len(seeds)))
    cell_draws = source_draws(files.dataset_group.to_numpy(), reps, rng)
    truth = {contract: labels[contract]["test"].loc[file_order].to_numpy(float) for contract in CONTRACTS}
    rows = []
    for model in MODELS:
        current = test[test.model.eq(model)]
        prediction = {contract: current[current.fit_contract.eq(contract)].pivot(index="seed", columns="file_name", values="predicted_eol").loc[seeds, file_order].to_numpy(float) for contract in CONTRACTS}
        for name in METRICS:
            boot = {}
            for fit_contract, evaluation_contract, cell in (("release", "release", "AA"), ("release", "current_generator", "AB"), ("current_generator", "release", "BA"), ("current_generator", "current_generator", "BB")):
                y = np.broadcast_to(truth[evaluation_contract], prediction[fit_contract].shape)
                boot[cell] = boot_seed_metrics(contribution(y, prediction[fit_contract], name), seed_draws, cell_draws, name == "rmse")
            rows.append(pd.DataFrame({
                "replicate": np.arange(reps),
                "model": model,
                "metric": name,
                "evaluation_contrast": np.median(((boot["AB"] - boot["AA"]) + (boot["BB"] - boot["BA"])) / 2, axis=1),
                "fitting_contrast": np.median(((boot["BA"] - boot["AA"]) + (boot["BB"] - boot["AB"])) / 2, axis=1),
                "interaction": np.median(boot["BB"] - boot["BA"] - boot["AB"] + boot["AA"], axis=1),
            }))
    replicates = pd.concat(rows, ignore_index=True)
    summary = []
    for (model, name), group in replicates.groupby(["model", "metric"], sort=True):
        for contrast in ("evaluation_contrast", "fitting_contrast", "interaction"):
            values = group[contrast]
            summary.append({
                "model": model,
                "metric": name,
                "contrast": contrast,
                "bootstrap_median": float(values.median()),
                "bootstrap_95_low": float(values.quantile(0.025)),
                "bootstrap_95_high": float(values.quantile(0.975)),
                "replicates": reps,
                "resampling": "seeds with replacement and test cells within source with replacement; four cell metrics and contrasts per seed, then seed median",
            })
    return replicates, pd.DataFrame(summary)


def shift_bootstrap(wide, reps):
    seeds = np.sort(wide.seed.unique())
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    pair_left, pair_right = np.triu_indices(len(seeds), k=1)
    rows = []
    for model in MODELS:
        current = wide[wide.model.eq(model)]
        release = current.pivot(index="seed", columns="file_name", values="release").loc[seeds]
        generated = current.pivot(index="seed", columns="file_name", values="current_generator").loc[seeds, release.columns]
        metadata = current.drop_duplicates("file_name").set_index("file_name").loc[release.columns]
        for status, mask in (("all", np.ones(len(metadata), dtype=bool)), ("changed", metadata.labels_differ.to_numpy(bool)), ("unchanged", ~metadata.labels_differ.to_numpy(bool))):
            left = release.to_numpy(float)[:, mask]
            right = generated.to_numpy(float)[:, mask]
            draws = source_draws(metadata.dataset_group.to_numpy()[mask], reps, rng)
            seed_draws = rng.integers(0, len(seeds), size=(reps, len(seeds)))
            cross = np.empty(reps)
            within_left = np.empty(reps)
            within_right = np.empty(reps)
            within_pooled = np.empty(reps)
            for start in range(0, reps, 25):
                stop = min(start + 25, reps)
                cells = draws[start:stop]
                drawn = seed_draws[start:stop]
                cross_values = np.abs(right[drawn[:, :, None], cells[:, None, :]] - left[drawn[:, :, None], cells[:, None, :]]).mean(axis=2)
                within_left_values = np.abs(left[drawn[:, pair_left, None], cells[:, None, :]] - left[drawn[:, pair_right, None], cells[:, None, :]]).mean(axis=2)
                within_right_values = np.abs(right[drawn[:, pair_left, None], cells[:, None, :]] - right[drawn[:, pair_right, None], cells[:, None, :]]).mean(axis=2)
                cross[start:stop] = np.median(cross_values, axis=1)
                within_left[start:stop] = np.median(within_left_values, axis=1)
                within_right[start:stop] = np.median(within_right_values, axis=1)
                within_pooled[start:stop] = np.median(np.concatenate([within_left_values, within_right_values], axis=1), axis=1)
            baseline = within_pooled
            rows.append(pd.DataFrame({
                "replicate": np.arange(reps),
                "model": model,
                "label_status": status,
                "cross_contract_mean_absolute_shift": cross,
                "within_release_mean_absolute_shift": within_left,
                "within_current_mean_absolute_shift": within_right,
                "within_pooled_pairwise_median": within_pooled,
                "cross_minus_within_baseline": cross - baseline,
                "cross_to_within_ratio": np.divide(cross, baseline, out=np.full_like(cross, np.nan), where=baseline > 0),
            }))
    replicates = pd.concat(rows, ignore_index=True)
    summary = []
    for (model, status), group in replicates.groupby(["model", "label_status"], sort=True):
        for quantity in ("cross_contract_mean_absolute_shift", "within_release_mean_absolute_shift", "within_current_mean_absolute_shift", "within_pooled_pairwise_median", "cross_minus_within_baseline", "cross_to_within_ratio"):
            values = group[quantity].replace([np.inf, -np.inf], np.nan).dropna()
            summary.append({
                "model": model,
                "label_status": status,
                "quantity": quantity,
                "bootstrap_median": float(values.median()) if len(values) else np.nan,
                "bootstrap_95_low": float(values.quantile(0.025)) if len(values) else np.nan,
                "bootstrap_95_high": float(values.quantile(0.975)) if len(values) else np.nan,
                "finite_replicates": len(values),
                "resampling": "paired seed units with replacement; all pairwise distances within each resampled seed ensemble; test cells within source with replacement",
            })
    return replicates, pd.DataFrame(summary)


def conformal_tables(predictions, labels, paired):
    metric_rows = []
    trace_rows = []
    paired_index = paired.set_index("file_name")
    pooled_train = pd.concat([labels[contract]["train"] for contract in CONTRACTS], ignore_index=True)
    common_train_iqr = float(pooled_train.quantile(0.75) - pooled_train.quantile(0.25))
    for contract in CONTRACTS:
        train_iqr = float(labels[contract]["train"].quantile(0.75) - labels[contract]["train"].quantile(0.25))
        for model in MODELS:
            for seed in SEEDS:
                fit = predictions[(predictions.fit_contract.eq(contract)) & (predictions.model.eq(model)) & (predictions.seed.eq(seed))]
                calibration = fit[fit.split.eq("calibration")].copy().merge(labels[contract]["calibration"].rename("eol_cycle"), left_on="file_name", right_index=True, validate="one_to_one")
                calibration["absolute_residual"] = np.abs(calibration.eol_cycle - calibration.predicted_eol)
                test = fit[fit.split.eq("test")].copy().merge(labels[contract]["test"].rename("eol_cycle"), left_on="file_name", right_index=True, validate="one_to_one")
                for nominal in NOMINALS:
                    for ablation, excluded in (("none", None), ("drop_ISU_ILCC", "ISU_ILCC"), ("drop_Tongji", "Tongji")):
                        pool = calibration if excluded is None else calibration[~calibration.dataset_group.eq(excluded)]
                        q, k = finite_quantile(pool.absolute_residual, nominal)
                        lower = test.predicted_eol.to_numpy(float) - q
                        upper = test.predicted_eol.to_numpy(float) + q
                        truth = test.eol_cycle.to_numpy(float)
                        scored = test.assign(lower=lower, upper=upper, width=2 * q, covered=(truth >= lower) & (truth <= upper), interval_score=interval_score(truth, lower, upper, nominal))
                        populations = [("ALL", scored), *[(source, group) for source, group in scored.groupby("dataset_group", sort=True)]]
                        for population, group in populations:
                            metric_rows.append({
                                "contract": contract,
                                "model": model,
                                "seed": int(seed),
                                "nominal": nominal,
                                "calibration_ablation": ablation,
                                "excluded_calibration_source": excluded or "",
                                "population": population,
                                "n_calibration": len(pool),
                                "order_statistic": k,
                                "calibration_quantile": q,
                                "n_test": len(group),
                                "coverage": float(group.covered.mean()),
                                "mean_width": float(group.width.mean()),
                                "median_width": float(group.width.median()),
                                "normalized_mean_width_train_iqr": float(group.width.mean() / train_iqr),
                                "normalized_mean_width_common_train_iqr": float(group.width.mean() / common_train_iqr),
                                "mean_interval_score": float(group.interval_score.mean()),
                                "median_interval_score": float(group.interval_score.median()),
                                "guarantee_scope": (
                                    "pooled marginal under exchangeability assumptions"
                                    if ablation == "none" and population == "ALL"
                                    else "empirical source-subgroup diagnostic; no subgroup guarantee"
                                    if ablation == "none"
                                    else "composition sensitivity only; no retained-sample guarantee"
                                ),
                            })
                    ordered = calibration.sort_values(["absolute_residual", "file_name"]).reset_index(drop=True)
                    q, k = finite_quantile(ordered.absolute_residual, nominal)
                    ties = int(np.isclose(ordered.absolute_residual, q, rtol=0, atol=1e-12).sum())
                    for rank in range(max(1, k - 2), min(len(ordered), k + 2) + 1):
                        row = ordered.iloc[rank - 1]
                        meta = paired_index.loc[row.file_name]
                        trace_rows.append({
                            "contract": contract,
                            "model": model,
                            "seed": int(seed),
                            "nominal": nominal,
                            "n_calibration": len(ordered),
                            "order_statistic": k,
                            "rank": rank,
                            "rank_offset": rank - k,
                            "file_name": row.file_name,
                            "dataset_group": row.dataset_group,
                            "eol_cycle": row.eol_cycle,
                            "predicted_eol": row.predicted_eol,
                            "absolute_residual": row.absolute_residual,
                            "calibration_quantile": q,
                            "residual_minus_quantile": row.absolute_residual - q,
                            "labels_differ": bool(meta.labels_differ),
                            "label_delta_cycles": float(meta.label_delta_cycles),
                            "ties_at_quantile": ties,
                        })
    metrics = pd.DataFrame(metric_rows)
    traces = pd.DataFrame(trace_rows)
    primary = metrics[(metrics.calibration_ablation.eq("none")) & (metrics.population.eq("ALL"))]
    differences = []
    for (model, seed, nominal), group in primary.groupby(["model", "seed", "nominal"], sort=True):
        values = group.set_index("contract")
        differences.append({
            "model": model,
            "seed": int(seed),
            "nominal": nominal,
            **{f"B_minus_A_{name}": float(values.loc["current_generator", name] - values.loc["release", name]) for name in ("coverage", "mean_width", "normalized_mean_width_train_iqr", "normalized_mean_width_common_train_iqr", "mean_interval_score")},
        })
    differences = pd.DataFrame(differences)
    summary = []
    for (model, nominal), group in differences.groupby(["model", "nominal"], sort=True):
        for name in ("B_minus_A_coverage", "B_minus_A_mean_width", "B_minus_A_normalized_mean_width_train_iqr", "B_minus_A_normalized_mean_width_common_train_iqr", "B_minus_A_mean_interval_score"):
            summary.append({
                "model": model,
                "nominal": nominal,
                "quantity": name,
                "seed_median": float(group[name].median()),
                "seed_2p5": float(group[name].quantile(0.025)),
                "seed_97p5": float(group[name].quantile(0.975)),
                "positive_seed_count": int((group[name] > 0).sum()),
                "negative_seed_count": int((group[name] < 0).sum()),
                "zero_seed_count": int(np.isclose(group[name], 0).sum()),
                "n_seeds": len(group),
            })
    return metrics, traces, differences, pd.DataFrame(summary)


def training_summary(runs):
    records = pd.DataFrame(runs)
    rows = []
    for (model, contract), group in records.groupby(["model", "fit_contract"], sort=True):
        completed = group[group.status.eq("completed")]
        final_loss = pd.to_numeric(completed["final_loss"], errors="coerce").dropna() if "final_loss" in completed else pd.Series(dtype=float)
        epochs = pd.to_numeric(completed["epochs"], errors="coerce").dropna() if "epochs" in completed else pd.Series(dtype=float)
        rows.append({
            "model": model,
            "fit_contract": contract,
            "attempts": len(group),
            "completed": len(completed),
            "failed": int(group.status.eq("failed").sum()),
            "median_seconds": float(completed.seconds.median()),
            "min_seconds": float(completed.seconds.min()),
            "max_seconds": float(completed.seconds.max()),
            "fixed_epochs": int(epochs.iloc[0]) if len(epochs) and epochs.nunique() == 1 else np.nan,
            "median_final_training_loss": float(final_loss.median()) if len(final_loss) else np.nan,
            "min_final_training_loss": float(final_loss.min()) if len(final_loss) else np.nan,
            "max_final_training_loss": float(final_loss.max()) if len(final_loss) else np.nan,
        })
    return pd.DataFrame(rows)
