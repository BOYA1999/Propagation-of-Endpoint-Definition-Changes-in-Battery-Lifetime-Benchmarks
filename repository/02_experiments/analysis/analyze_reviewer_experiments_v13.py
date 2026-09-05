import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


CONTRACTS = ("release", "current_generator")
MODELS = ("ridge", "xgboost", "cpmlp", "cptransformer")
METRICS = ("mape", "smape", "mae", "rmse", "log_mae")
NOMINALS = (0.9, 0.95)
BOOTSTRAP_SEED = 130904


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric(y, p, name):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    error = np.abs(y - p)
    if name == "mape":
        return float(np.mean(error / y))
    if name == "smape":
        return float(np.mean(2 * error / (np.abs(y) + np.abs(p))))
    if name == "mae":
        return float(np.mean(error))
    if name == "rmse":
        return float(np.sqrt(np.mean((y - p) ** 2)))
    return float(np.mean(np.abs(np.log(y) - np.log(p))))


def metric_contribution(y, p, name):
    error = np.abs(y - p)
    if name == "mape":
        return error / y
    if name == "smape":
        return 2 * error / (np.abs(y) + np.abs(p))
    if name == "mae":
        return error
    if name == "rmse":
        return (y - p) ** 2
    return np.abs(np.log(y) - np.log(p))


def interval_score(y, lower, upper, nominal):
    alpha = 1 - nominal
    score = upper - lower
    score = score + np.where(y < lower, 2 * (lower - y) / alpha, 0)
    score = score + np.where(y > upper, 2 * (y - upper) / alpha, 0)
    return score


def finite_quantile(values, nominal):
    values = np.sort(np.asarray(values, dtype=float))
    k = int(np.ceil((len(values) + 1) * nominal))
    if k > len(values):
        return np.nan, k
    return float(values[k - 1]), k


def load_labels(root):
    result = {}
    for contract in CONTRACTS:
        result[contract] = {}
        for split, name in (("train", "train_labels.csv"), ("calibration", "calibration_labels.csv"), ("test", "test_labels_sealed.csv")):
            frame = pd.read_csv(root / contract / name)
            if frame.file_name.duplicated().any():
                raise SystemExit(f"Duplicate labels: {contract}/{split}")
            result[contract][split] = frame.set_index("file_name").eol_cycle.astype(float)
    return result


def reveal(predictions, labels, split, evaluation_contract):
    frame = predictions[predictions.split.eq(split)].copy()
    truth = labels[evaluation_contract][split].rename("eol_cycle")
    frame = frame.merge(truth, left_on="file_name", right_index=True, how="left", validate="many_to_one")
    if frame.eol_cycle.isna().any():
        raise SystemExit(f"Missing {split} labels for {evaluation_contract}")
    frame["evaluation_contract"] = evaluation_contract
    return frame


def point_outputs(predictions, labels):
    revealed = pd.concat([reveal(predictions, labels, "test", contract) for contract in CONTRACTS], ignore_index=True)
    rows = []
    for keys, group in revealed.groupby(["fit_contract", "evaluation_contract", "model", "seed"], sort=True):
        row = {
            "fit_contract": keys[0],
            "evaluation_contract": keys[1],
            "model": keys[2],
            "seed": int(keys[3]),
            "n_test": len(group),
        }
        for name in METRICS:
            row[name] = metric(group.eol_cycle, group.predicted_eol, name)
        rows.append(row)
    metrics = pd.DataFrame(rows)
    factors = []
    for (model, seed), group in metrics.groupby(["model", "seed"], sort=True):
        cells = {(row.fit_contract, row.evaluation_contract): row for row in group.itertuples()}
        aa = cells[("release", "release")]
        ab = cells[("release", "current_generator")]
        ba = cells[("current_generator", "release")]
        bb = cells[("current_generator", "current_generator")]
        for name in METRICS:
            values = {"AA": getattr(aa, name), "AB": getattr(ab, name), "BA": getattr(ba, name), "BB": getattr(bb, name)}
            factors.append({
                "model": model,
                "seed": int(seed),
                "metric": name,
                **values,
                "evaluation_contrast": ((values["AB"] - values["AA"]) + (values["BB"] - values["BA"])) / 2,
                "fitting_contrast": ((values["BA"] - values["AA"]) + (values["BB"] - values["AB"])) / 2,
                "interaction": values["BB"] - values["BA"] - values["AB"] + values["AA"],
            })
    return revealed, metrics, pd.DataFrame(factors)


def ranking_outputs(point_metrics):
    cell_map = {
        ("release", "release"): "AA",
        ("release", "current_generator"): "AB",
        ("current_generator", "release"): "BA",
        ("current_generator", "current_generator"): "BB",
    }
    rows = []
    for record in point_metrics.itertuples():
        cell = cell_map[(record.fit_contract, record.evaluation_contract)]
        for name in METRICS:
            rows.append({"cell": cell, "model": record.model, "seed": int(record.seed), "metric": name, "score": getattr(record, name)})
    ranks = pd.DataFrame(rows)
    ranks["rank"] = ranks.groupby(["cell", "seed", "metric"])["score"].rank(method="min")
    comparisons = []
    directions = (("AA", "AB", "evaluation_Afit"), ("BA", "BB", "evaluation_Bfit"), ("AA", "BA", "fitting_Aeval"), ("AB", "BB", "fitting_Beval"), ("AA", "BB", "matched_contract"))
    for seed in sorted(ranks.seed.unique()):
        for name in METRICS:
            current = ranks[(ranks.seed.eq(seed)) & (ranks.metric.eq(name))]
            for left, right, direction in directions:
                a = current[current.cell.eq(left)].sort_values("model").set_index("model")
                b = current[current.cell.eq(right)].sort_values("model").set_index("model")
                order_a = a.reset_index().sort_values(["score", "model"]).model.tolist()
                order_b = b.reset_index().sort_values(["score", "model"]).model.tolist()
                comparisons.append({
                    "seed": int(seed),
                    "metric": name,
                    "direction": direction,
                    "left_cell": left,
                    "right_cell": right,
                    "spearman_rank": float(spearmanr(a["rank"], b["rank"]).statistic),
                    "kendall_rank": float(kendalltau(a["rank"], b["rank"]).statistic),
                    "full_order_same": order_a == order_b,
                    "best_model_left": order_a[0],
                    "best_model_right": order_b[0],
                    "best_model_changed": order_a[0] != order_b[0],
                    "top2_set_same": set(order_a[:2]) == set(order_b[:2]),
                    "order_left": ">".join(order_a),
                    "order_right": ">".join(order_b),
                })
    comparisons = pd.DataFrame(comparisons)
    matched = ranks[ranks.cell.isin(["AA", "BB"])].copy()
    best_counts = matched[matched["rank"].eq(1)].groupby(["cell", "metric", "model"], as_index=False).size().rename(columns={"size": "best_seed_count"})
    best_grid = pd.MultiIndex.from_product(
        [["AA", "BB"], METRICS, MODELS], names=["cell", "metric", "model"]
    ).to_frame(index=False)
    best = best_grid.merge(best_counts, on=["cell", "metric", "model"], how="left")
    best["best_seed_count"] = best.best_seed_count.fillna(0).astype(int)
    best["total_seeds"] = matched.seed.nunique()
    best["best_seed_fraction"] = best.best_seed_count / best.total_seeds
    best["tie_handling"] = "exact rank-1 ties count as co-winners"
    pairwise = []
    for (cell, name), group in matched.groupby(["cell", "metric"], sort=True):
        wide = group.pivot(index="seed", columns="model", values="score")
        for left, right in itertools.combinations(MODELS, 2):
            delta = wide[left] - wide[right]
            pairwise.append({
                "cell": cell,
                "metric": name,
                "model_left": left,
                "model_right": right,
                "left_wins": int((delta < -1e-12).sum()),
                "right_wins": int((delta > 1e-12).sum()),
                "ties": int((np.abs(delta) <= 1e-12).sum()),
                "median_left_minus_right": float(delta.median()),
            })
    return ranks, comparisons, best, pd.DataFrame(pairwise)


def paired_metadata(path):
    frame = pd.read_csv(path)
    frame["labels_differ"] = frame.labels_differ.astype(str).str.lower().eq("true")
    return frame


def refit_variability(predictions, paired):
    test = predictions[predictions.split.eq("test")].merge(
        paired[["file_name", "labels_differ", "label_delta_cycles"]], on="file_name", validate="many_to_one"
    )
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
    summary = []
    for model in MODELS:
        for status in ("all", "changed", "unchanged"):
            c = cross[(cross.model.eq(model)) & (cross.label_status.eq(status))].mean_absolute_shift
            a = within[(within.model.eq(model)) & (within.fit_contract.eq("release")) & (within.label_status.eq(status))].mean_absolute_shift
            b = within[(within.model.eq(model)) & (within.fit_contract.eq("current_generator")) & (within.label_status.eq(status))].mean_absolute_shift
            pooled = pd.concat([a, b], ignore_index=True)
            baseline = float(pooled.median())
            summary.append({
                "model": model,
                "label_status": status,
                "cross_contract_seed_median_mean_absolute_shift": float(c.median()),
                "within_release_pair_median_mean_absolute_shift": float(a.median()),
                "within_current_pair_median_mean_absolute_shift": float(b.median()),
                "within_pooled_pair_median_mean_absolute_shift": baseline,
                "cross_to_within_ratio": float(c.median() / baseline) if baseline > 0 else np.nan,
                "cross_seed_fraction_above_within_median": float((c > baseline).mean()),
                "cross_seed_count": len(c),
                "within_pair_count_per_contract": len(a),
            })
    return wide, cross, within, pd.DataFrame(summary)


def source_stratified_draws(sources, reps, rng):
    parts = []
    sources = np.asarray(sources)
    for source in sorted(set(sources)):
        indices = np.flatnonzero(sources == source)
        parts.append(rng.choice(indices, size=(reps, len(indices)), replace=True))
    return np.concatenate(parts, axis=1)


def bootstrap_seed_metrics(values, seed_draws, cell_draws, rmse=False, chunk=100):
    output = np.empty(seed_draws.shape, dtype=float)
    for start in range(0, len(seed_draws), chunk):
        stop = min(start + chunk, len(seed_draws))
        selected = values[seed_draws[start:stop, :, None], cell_draws[start:stop, None, :]]
        per_seed = selected.mean(axis=2)
        if rmse:
            per_seed = np.sqrt(per_seed)
        output[start:stop] = per_seed
    return output


def factor_bootstrap(predictions, labels, reps):
    test = predictions[predictions.split.eq("test")]
    seeds = np.sort(test.seed.unique())
    files = test[test.model.eq(MODELS[0])].drop_duplicates("file_name").sort_values("file_name")
    file_order = files.file_name.tolist()
    sources = files.dataset_group.to_numpy()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    seed_draws = rng.integers(0, len(seeds), size=(reps, len(seeds)))
    cell_draws = source_stratified_draws(sources, reps, rng)
    truth = {contract: labels[contract]["test"].loc[file_order].to_numpy(float) for contract in CONTRACTS}
    rows = []
    for model in MODELS:
        model_frame = test[test.model.eq(model)]
        prediction = {
            contract: model_frame[model_frame.fit_contract.eq(contract)].pivot(index="seed", columns="file_name", values="predicted_eol").loc[seeds, file_order].to_numpy(float)
            for contract in CONTRACTS
        }
        for name in METRICS:
            boot = {}
            for fit_contract, evaluation_contract, cell in (
                ("release", "release", "AA"),
                ("release", "current_generator", "AB"),
                ("current_generator", "release", "BA"),
                ("current_generator", "current_generator", "BB"),
            ):
                y = np.broadcast_to(truth[evaluation_contract], prediction[fit_contract].shape)
                contribution = metric_contribution(y, prediction[fit_contract], name)
                boot[cell] = bootstrap_seed_metrics(contribution, seed_draws, cell_draws, name == "rmse")
            evaluation = np.median(((boot["AB"] - boot["AA"]) + (boot["BB"] - boot["BA"])) / 2, axis=1)
            fitting = np.median(((boot["BA"] - boot["AA"]) + (boot["BB"] - boot["AB"])) / 2, axis=1)
            interaction = np.median(boot["BB"] - boot["BA"] - boot["AB"] + boot["AA"], axis=1)
            rows.append(pd.DataFrame({
                "replicate": np.arange(reps),
                "model": model,
                "metric": name,
                "evaluation_contrast": evaluation,
                "fitting_contrast": fitting,
                "interaction": interaction,
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


def prediction_shift_bootstrap(wide, reps):
    seeds = np.sort(wide.seed.unique())
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    pair_left, pair_right = np.triu_indices(len(seeds), k=1)
    rows = []
    for model in MODELS:
        current = wide[wide.model.eq(model)]
        release = current.pivot(index="seed", columns="file_name", values="release").loc[seeds]
        generated = current.pivot(index="seed", columns="file_name", values="current_generator").loc[seeds, release.columns]
        metadata = current.drop_duplicates("file_name").set_index("file_name").loc[release.columns]
        for status, status_mask in (("all", np.ones(len(metadata), dtype=bool)), ("changed", metadata.labels_differ.to_numpy(bool)), ("unchanged", ~metadata.labels_differ.to_numpy(bool))):
            a = release.to_numpy(float)[:, status_mask]
            b = generated.to_numpy(float)[:, status_mask]
            sources = metadata.dataset_group.to_numpy()[status_mask]
            cell_draws = source_stratified_draws(sources, reps, rng)
            seed_draws = rng.integers(0, len(seeds), size=(reps, len(seeds)))
            cross = np.empty(reps)
            within_a = np.empty(reps)
            within_b = np.empty(reps)
            within_pooled = np.empty(reps)
            for start in range(0, reps, 25):
                stop = min(start + 25, reps)
                cells = cell_draws[start:stop]
                drawn = seed_draws[start:stop]
                cross_values = np.abs(
                    b[drawn[:, :, None], cells[:, None, :]]
                    - a[drawn[:, :, None], cells[:, None, :]]
                ).mean(axis=2)
                within_a_values = np.abs(
                    a[drawn[:, pair_left, None], cells[:, None, :]]
                    - a[drawn[:, pair_right, None], cells[:, None, :]]
                ).mean(axis=2)
                within_b_values = np.abs(
                    b[drawn[:, pair_left, None], cells[:, None, :]]
                    - b[drawn[:, pair_right, None], cells[:, None, :]]
                ).mean(axis=2)
                cross[start:stop] = np.median(cross_values, axis=1)
                within_a[start:stop] = np.median(within_a_values, axis=1)
                within_b[start:stop] = np.median(within_b_values, axis=1)
                within_pooled[start:stop] = np.median(np.concatenate([within_a_values, within_b_values], axis=1), axis=1)
            baseline = within_pooled
            rows.append(pd.DataFrame({
                "replicate": np.arange(reps),
                "model": model,
                "label_status": status,
                "cross_contract_mean_absolute_shift": cross,
                "within_release_mean_absolute_shift": within_a,
                "within_current_mean_absolute_shift": within_b,
                "within_pooled_pairwise_median": within_pooled,
                "cross_minus_within_baseline": cross - baseline,
                "cross_to_within_ratio": np.divide(cross, baseline, out=np.full_like(cross, np.nan), where=baseline > 0),
            }))
    replicates = pd.concat(rows, ignore_index=True)
    summary = []
    for (model, status), group in replicates.groupby(["model", "label_status"], sort=True):
        for value in ("cross_contract_mean_absolute_shift", "within_release_mean_absolute_shift", "within_current_mean_absolute_shift", "within_pooled_pairwise_median", "cross_minus_within_baseline", "cross_to_within_ratio"):
            finite = group[value].replace([np.inf, -np.inf], np.nan).dropna()
            summary.append({
                "model": model,
                "label_status": status,
                "quantity": value,
                "bootstrap_median": float(finite.median()) if len(finite) else np.nan,
                "bootstrap_95_low": float(finite.quantile(0.025)) if len(finite) else np.nan,
                "bootstrap_95_high": float(finite.quantile(0.975)) if len(finite) else np.nan,
                "finite_replicates": len(finite),
                "resampling": "paired seed units with replacement; all pairwise distances within each resampled seed ensemble; test cells within source with replacement",
            })
    return replicates, pd.DataFrame(summary)


def conformal_outputs(predictions, labels, paired):
    metrics = []
    traces = []
    paired_index = paired.set_index("file_name")
    pooled_train = pd.concat([labels[contract]["train"] for contract in CONTRACTS], ignore_index=True)
    common_train_iqr = float(pooled_train.quantile(0.75) - pooled_train.quantile(0.25))
    for contract in CONTRACTS:
        train_iqr = float(labels[contract]["train"].quantile(0.75) - labels[contract]["train"].quantile(0.25))
        for model in MODELS:
            for seed in sorted(predictions.seed.unique()):
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
                        scored = test.assign(
                            lower=lower,
                            upper=upper,
                            width=2 * q,
                            covered=(truth >= lower) & (truth <= upper),
                            interval_score=interval_score(truth, lower, upper, nominal),
                        )
                        for population, group in [("ALL", scored), *[(source, part) for source, part in scored.groupby("dataset_group", sort=True)]]:
                            metrics.append({
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
                    tie_count = int(np.isclose(ordered.absolute_residual, q, rtol=0, atol=1e-12).sum())
                    for rank in range(max(1, k - 2), min(len(ordered), k + 2) + 1):
                        row = ordered.iloc[rank - 1]
                        meta = paired_index.loc[row.file_name]
                        traces.append({
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
                            "ties_at_quantile": tie_count,
                        })
    metrics = pd.DataFrame(metrics)
    traces = pd.DataFrame(traces)
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


def original_xgb_comparison(release_path, current_path, labels, controlled_factors):
    predictions = []
    for contract, path in (("release", release_path), ("current_generator", current_path)):
        frame = pd.read_csv(path)
        frame = frame[(frame.split.eq("test")) & (frame.model.eq("xgboost"))].copy()
        frame["fit_contract"] = contract
        predictions.append(frame[["fit_contract", "split", "dataset_group", "chemistry_system", "file_name", "model", "seed", "predicted_eol"]])
    _, _, original = point_outputs(pd.concat(predictions, ignore_index=True), labels)
    original["protocol"] = "original_end_to_end_contract_specific_cv"
    controlled = controlled_factors[(controlled_factors.model.eq("xgboost")) & (controlled_factors.seed.isin(original.seed))].copy()
    controlled["protocol"] = "fixed_label_only_control_v13"
    return pd.concat([original, controlled], ignore_index=True)


def training_summary(path):
    records = pd.DataFrame(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
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
    return records, pd.DataFrame(rows)


def validate_campaign_inputs(predictions, training_records, frozen, labels, paired):
    required_prediction_columns = {
        "fit_contract", "split", "dataset_group", "chemistry_system", "file_name",
        "model", "seed", "predicted_eol", "prediction_kind",
    }
    required_log_columns = {"fit_contract", "model", "seed", "status", "rows", "state_sha256"}
    if not required_prediction_columns.issubset(predictions.columns) or not required_log_columns.issubset(training_records.columns):
        raise SystemExit("Controlled inputs lack required columns")

    seeds = [int(seed) for seed in frozen.get("seeds", [])]
    contracts = frozen.get("contracts", [])
    models = frozen.get("models", [])
    split_counts = frozen.get("split_counts", {})
    if not seeds or len(seeds) != len(set(seeds)):
        raise SystemExit("Frozen configuration has an invalid seed list")
    if set(contracts) != set(CONTRACTS) or len(contracts) != len(CONTRACTS):
        raise SystemExit("Frozen contract set does not match the analysis contract")
    if set(models) != set(MODELS) or len(models) != len(MODELS):
        raise SystemExit("Frozen model set does not match the analysis contract")
    if set(predictions.seed.astype(int)) != set(seeds):
        raise SystemExit("Prediction seeds do not match the frozen configuration")
    if set(predictions.fit_contract) != set(CONTRACTS) or set(predictions.model) != set(MODELS):
        raise SystemExit("Prediction contracts or models are incomplete")
    if set(predictions.split) != {"calibration", "test"}:
        raise SystemExit("Prediction splits are incomplete or unexpected")
    if set(predictions.prediction_kind) != {"train_refit_blind_fixed_control_v13"}:
        raise SystemExit("Unexpected prediction provenance tag")

    key = ["fit_contract", "split", "file_name", "model", "seed"]
    if predictions.duplicated(key).any() or not np.isfinite(predictions.predicted_eol).all() or not (predictions.predicted_eol > 0).all():
        raise SystemExit("Controlled predictions contain duplicate, non-finite, or non-positive values")
    for split in ("train", "calibration", "test"):
        expected_count = int(split_counts.get(split, -1))
        for contract in CONTRACTS:
            if len(labels[contract][split]) != expected_count:
                raise SystemExit(f"Frozen split count mismatch: {contract}/{split}")
        if set(labels[CONTRACTS[0]][split].index) != set(labels[CONTRACTS[1]][split].index):
            raise SystemExit(f"Contract file-ID sets differ for split: {split}")

    if paired.file_name.duplicated().any() or set(paired.split) != {"train", "calibration", "test"}:
        raise SystemExit("Paired-contract table has duplicate IDs or invalid splits")
    expected_all_ids = set().union(*(set(labels[CONTRACTS[0]][split].index) for split in ("train", "calibration", "test")))
    if set(paired.file_name) != expected_all_ids:
        raise SystemExit("Paired-contract file IDs do not match the label files")
    paired_index = paired.set_index("file_name")
    for split in ("train", "calibration", "test"):
        expected_ids = set(labels[CONTRACTS[0]][split].index)
        paired_split = paired[paired.split.eq(split)].set_index("file_name")
        if set(paired_split.index) != expected_ids:
            raise SystemExit(f"Paired-contract split mapping mismatch: {split}")
        release_truth = labels["release"][split].loc[paired_split.index].to_numpy(float)
        current_truth = labels["current_generator"][split].loc[paired_split.index].to_numpy(float)
        if (
            not np.allclose(paired_split.release_eol.to_numpy(float), release_truth, rtol=0, atol=1e-12)
            or not np.allclose(paired_split.current_generator_eol.to_numpy(float), current_truth, rtol=0, atol=1e-12)
            or not np.allclose(paired_split.label_delta_cycles.to_numpy(float), current_truth - release_truth, rtol=0, atol=1e-12)
            or not np.array_equal(paired_split.labels_differ.to_numpy(bool), current_truth != release_truth)
        ):
            raise SystemExit(f"Paired-contract values do not reproduce the label files: {split}")
    test_paired = paired[paired.split.eq("test")]
    if int(test_paired.labels_differ.sum()) != 44 or int((~test_paired.labels_differ).sum()) != 114:
        raise SystemExit("Unexpected changed/unchanged test-cell counts")

    prediction_metadata = predictions[["file_name", "split", "dataset_group", "chemistry_system"]].drop_duplicates()
    if prediction_metadata.file_name.duplicated().any():
        raise SystemExit("Prediction split/source metadata vary across fitted runs")
    prediction_metadata = prediction_metadata.set_index("file_name")
    expected_prediction_ids = set(labels["release"]["calibration"].index) | set(labels["release"]["test"].index)
    if set(prediction_metadata.index) != expected_prediction_ids:
        raise SystemExit("Prediction metadata IDs are incomplete")
    reference_metadata = paired_index.loc[prediction_metadata.index]
    if not prediction_metadata.split.equals(reference_metadata.split) or not prediction_metadata.dataset_group.equals(reference_metadata.dataset_group):
        raise SystemExit("Prediction split/source metadata do not match the paired-contract table")

    for contract, model, seed, split in itertools.product(CONTRACTS, MODELS, seeds, ("calibration", "test")):
        group = predictions[
            predictions.fit_contract.eq(contract)
            & predictions.model.eq(model)
            & predictions.seed.eq(seed)
            & predictions.split.eq(split)
        ]
        if set(group.file_name) != set(labels[contract][split].index) or len(group) != int(split_counts[split]):
            raise SystemExit(f"Incomplete prediction cell: {contract}/{model}/{seed}/{split}")

    expected_runs = set(itertools.product(CONTRACTS, MODELS, seeds))
    actual_runs = list(zip(training_records.fit_contract, training_records.model, training_records.seed.astype(int)))
    if (
        len(training_records) != len(expected_runs)
        or len(actual_runs) != len(set(actual_runs))
        or set(actual_runs) != expected_runs
        or not training_records.status.eq("completed").all()
        or not training_records.rows.eq(int(split_counts["calibration"]) + int(split_counts["test"])).all()
        or not training_records.state_sha256.astype(str).str.fullmatch(r"[0-9a-f]{64}").all()
    ):
        raise SystemExit("Training log is incomplete, duplicated, failed, or inconsistent with the frozen campaign")
    return sorted(seeds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--frozen-configuration", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--paired-labels", type=Path, required=True)
    parser.add_argument("--original-release-predictions", type=Path, required=True)
    parser.add_argument("--original-current-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(args.predictions)
    labels = load_labels(args.labels_root)
    paired = paired_metadata(args.paired_labels)
    frozen = json.loads(args.frozen_configuration.read_text(encoding="utf-8"))
    training_records, run_summary = training_summary(args.training_log)
    seeds = validate_campaign_inputs(predictions, training_records, frozen, labels, paired)

    revealed, point_metrics, factors = point_outputs(predictions, labels)
    ranks, rank_comparisons, best, pairwise = ranking_outputs(point_metrics)
    shift_cells, cross, within, variability = refit_variability(predictions, paired)
    factor_replicates, factor_summary = factor_bootstrap(predictions, labels, args.bootstrap_reps)
    shift_replicates, shift_summary = prediction_shift_bootstrap(shift_cells, args.bootstrap_reps)
    conformal, quantile_trace, conformal_differences, conformal_summary = conformal_outputs(predictions, labels, paired)
    xgb_comparison = original_xgb_comparison(args.original_release_predictions, args.original_current_predictions, labels, factors)
    outputs = {
        "revealed_test_predictions.csv": revealed,
        "point_metrics_by_seed.csv": point_metrics,
        "factor_contrasts_by_seed.csv": factors,
        "model_ranks_by_cell.csv": ranks,
        "ranking_comparisons.csv": rank_comparisons,
        "best_model_frequency.csv": best,
        "pairwise_model_wins.csv": pairwise,
        "prediction_shift_cells.csv": shift_cells,
        "cross_contract_prediction_shift_by_seed.csv": cross,
        "within_contract_refit_pairwise.csv": within,
        "refit_variability_summary.csv": variability,
        "factor_seed_cell_bootstrap_replicates.csv": factor_replicates,
        "factor_seed_cell_bootstrap_summary.csv": factor_summary,
        "prediction_shift_bootstrap_replicates.csv": shift_replicates,
        "prediction_shift_bootstrap_summary.csv": shift_summary,
        "conformal_metrics.csv": conformal,
        "calibration_quantile_neighbourhood.csv": quantile_trace,
        "conformal_paired_differences_by_seed.csv": conformal_differences,
        "conformal_paired_summary.csv": conformal_summary,
        "xgboost_pipeline_vs_controlled.csv": xgb_comparison,
        "training_run_summary.csv": run_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output / name, index=False)

    matched = rank_comparisons[rank_comparisons.direction.eq("matched_contract")]
    primary_conformal = conformal_summary[(conformal_summary.nominal.eq(0.9))]
    xgb_mape = {key: float(value) for key, value in xgb_comparison[xgb_comparison.metric.eq("mape")].groupby("protocol").fitting_contrast.median().items()}
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ANALYSIS_COMPLETE_PENDING_INDEPENDENT_VALIDATION",
        "seeds": [int(seed) for seed in seeds],
        "n_seeds": len(seeds),
        "models": list(MODELS),
        "test_cells": 158,
        "calibration_cells": 162,
        "bootstrap_replicates": args.bootstrap_reps,
        "matched_rank_full_order_same": int(matched.full_order_same.sum()),
        "matched_rank_comparisons": len(matched),
        "matched_best_model_changes": int(matched.best_model_changed.sum()),
        "xgboost_mape_fitting_contrast_seed_median_by_protocol": xgb_mape,
        "primary_conformal_rows": len(primary_conformal),
        "training_attempts": len(training_records),
        "training_failures": int(training_records.status.eq("failed").sum()),
        "input_completeness": "exact frozen seed-contract-model-split Cartesian product and file-ID sets validated",
        "xgboost_pairing_boundary": "fixed hyperparameters and random_state, but XGBoost 3.3 default base_score is target-adaptive and therefore not a strict same-initial-intercept comparison",
        "interpretation_boundary": "controlled sensitivity analysis; neither endpoint is ground truth; source-removal conformal rows are composition diagnostics only",
    }
    (args.output / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha256(Path(__file__)),
        "input_sha256": {
            "predictions": sha256(args.predictions),
            "training_log": sha256(args.training_log),
            "frozen_configuration": sha256(args.frozen_configuration),
            "paired_labels": sha256(args.paired_labels),
            "original_release_predictions": sha256(args.original_release_predictions),
            "original_current_predictions": sha256(args.original_current_predictions),
            **{
                f"{contract}_{name}": sha256(args.labels_root / contract / name)
                for contract in CONTRACTS
                for name in ("train_labels.csv", "calibration_labels.csv", "test_labels_sealed.csv")
            },
        },
        "output_sha256": {name: sha256(args.output / name) for name in outputs},
        "analysis_summary_sha256": sha256(args.output / "analysis_summary.json"),
    }
    (args.output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
