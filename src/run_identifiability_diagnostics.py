import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 20260901
PERMUTATIONS = 1000


def sha256(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def main():
    global SEED, PERMUTATIONS
    parser = argparse.ArgumentParser(description="Run the ISU early-feature identifiability diagnostics.")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    args = parser.parse_args()
    SEED = args.seed
    PERMUTATIONS = args.permutations
    output = args.output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    feature_path = args.features
    label_path = args.labels
    features = pd.read_csv(feature_path)
    labels = pd.read_csv(label_path)
    frame = features.merge(labels[["file_name", "labels_differ", "label_delta_cycles"]], on="file_name", validate="one_to_one")
    frame = frame[frame.dataset_group.eq("ISU_ILCC")].copy()
    columns = [column for column in features.columns if column not in {"split", "dataset_group", "chemistry_system", "file_name"}]
    groups = {prefix: [column for column in columns if column.startswith(prefix)] for prefix in ("charge_capacity", "discharge_capacity", "coulombic_efficiency", "charge_voltage", "discharge_voltage", "charge_c_rate", "discharge_c_rate")}
    train = frame[frame.split.eq("train")]
    model = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LogisticRegression(C=1.0, l1_ratio=0, class_weight="balanced", solver="liblinear", random_state=SEED, max_iter=5000))])
    model.fit(train[columns], train.labels_differ.astype(int))
    rng = np.random.default_rng(SEED)
    importance_rows, correlation_rows = [], []
    for split in ("calibration", "test"):
        part = frame[frame.split.eq(split)].copy()
        y = part.labels_differ.astype(int).to_numpy()
        baseline_score = model.predict_proba(part[columns])[:, 1]
        baseline_auc = roc_auc_score(y, baseline_score)
        correlation_rows.append({"split": split, "subset": "all", "n_cells": len(part), "spearman_probability_vs_absolute_delta": float(spearmanr(baseline_score, part.label_delta_cycles.abs()).statistic)})
        changed = part.labels_differ.to_numpy(bool)
        correlation_rows.append({"split": split, "subset": "changed_only", "n_cells": int(changed.sum()), "spearman_probability_vs_absolute_delta": float(spearmanr(baseline_score[changed], part.loc[changed, "label_delta_cycles"].abs()).statistic)})
        for group, group_columns in groups.items():
            drops = []
            source = part[columns].copy()
            for _ in range(PERMUTATIONS):
                permuted = source.copy()
                order = rng.permutation(len(permuted))
                permuted.loc[:, group_columns] = source[group_columns].to_numpy()[order]
                drops.append(baseline_auc - roc_auc_score(y, model.predict_proba(permuted)[:, 1]))
            importance_rows.append({"split": split, "feature_group": group, "n_features": len(group_columns), "baseline_auroc": baseline_auc, "mean_auroc_drop": float(np.mean(drops)), "median_auroc_drop": float(np.median(drops)), "auroc_drop_95_low": float(np.quantile(drops, 0.025)), "auroc_drop_95_high": float(np.quantile(drops, 0.975)), "permutations": PERMUTATIONS})
    importance = pd.DataFrame(importance_rows)
    correlations = pd.DataFrame(correlation_rows)
    importance.to_csv(output / "isu_group_permutation_importance.csv", index=False)
    correlations.to_csv(output / "isu_probability_magnitude_correlation.csv", index=False)
    manifest = {"status": "PASS", "seed": SEED, "permutations": PERMUTATIONS, "input_sha256": {"features": sha256(feature_path), "labels": sha256(label_path)}, "outputs_sha256": {path.name: sha256(path) for path in output.iterdir() if path.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "importance": importance.sort_values(["split", "mean_auroc_drop"], ascending=[True, False]).groupby("split").head(3).to_dict("records"), "correlations": correlations.to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()
