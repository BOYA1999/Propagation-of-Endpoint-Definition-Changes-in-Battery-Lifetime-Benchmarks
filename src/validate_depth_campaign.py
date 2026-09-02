import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


parser = argparse.ArgumentParser(description="Validate the endpoint propagation analysis outputs.")
parser.add_argument("--labels", type=Path, required=True)
parser.add_argument("--core", type=Path, required=True)
parser.add_argument("--published-effects", type=Path, required=True)
parser.add_argument("--errors", type=Path, required=True)
parser.add_argument("--intervals", type=Path, required=True)
parser.add_argument("--auxiliary", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
CORE = args.core
RESULT = args.output

labels = pd.read_csv(args.labels)
wide = pd.read_csv(CORE / "cellwise_propagation.csv")
subgroups = pd.read_csv(CORE / "propagation_by_label_status.csv")
sources = pd.read_csv(CORE / "source_effect_contributions.csv")
metrics = pd.read_csv(CORE / "metric_factor_effects_by_seed.csv")
published = pd.read_csv(args.published_effects)
intervals = pd.read_csv(CORE / "interval_margin_decomposition_cells.csv")
transitions = pd.read_csv(CORE / "interval_coverage_transitions.csv")
leverage = pd.read_csv(CORE / "g34c2_training_leverage_cells.csv")
identity = pd.read_csv(CORE / "isu_early_identifiability_summary.csv")
manifest = json.loads((CORE / "manifest.json").read_text(encoding="utf-8"))
aux = args.auxiliary
importance = pd.read_csv(aux / "isu_group_permutation_importance.csv")
correlations = pd.read_csv(aux / "isu_probability_magnitude_correlation.csv")
aux_manifest = json.loads((aux / "manifest.json").read_text(encoding="utf-8"))

checks = {}
checks["paired_cohort_and_changed_count"] = len(labels) == 823 and int(labels.labels_differ.sum()) == 252
checks["cellwise_matrix_complete"] = len(wide) == 4 * 3 * 158 and set(wide.model) == {"ridge", "xgboost", "cpmlp", "cptransformer"}
checks["changed_test_count"] = int(wide[wide.model.eq("ridge") & wide.seed.eq(42)].labels_differ.sum()) == 44
checks["unchanged_evaluation_effect_exact_zero"] = float(wide.loc[~wide.labels_differ, "evaluation_effect"].abs().max()) < 1e-15

source_ok = True
subgroup_ok = True
for (model, seed), group in wide.groupby(["model", "seed"]):
    src = sources[sources.model.eq(model) & sources.seed.eq(seed)]
    sub = subgroups[subgroups.model.eq(model) & subgroups.seed.eq(seed)]
    for effect in ("evaluation", "fitting", "interaction"):
        source_ok &= np.isclose(src[f"{effect}_contribution_to_full_mean"].sum(), group[f"{effect}_effect" if effect != "interaction" else "interaction"].mean(), atol=1e-12)
        subgroup_ok &= np.isclose(sub[f"{effect}_contribution_to_full_mean"].sum(), group[f"{effect}_effect" if effect != "interaction" else "interaction"].mean(), atol=1e-12)
checks["source_contributions_reconcile"] = bool(source_ok)
checks["label_status_contributions_reconcile"] = bool(subgroup_ok)

mape = metrics[metrics.metric.eq("MAPE")].sort_values(["model", "seed"]).reset_index(drop=True)
reference = published.sort_values(["model", "seed"]).reset_index(drop=True)
mapping = {"AA": "mape_AA", "AB": "mape_AB", "BA": "mape_BA", "BB": "mape_BB", "evaluation_effect": "evaluation_main_effect", "fitting_effect": "fitting_main_effect", "interaction": "interaction"}
checks["mape_reproduces_v4"] = all(np.allclose(mape[left], reference[right], atol=1e-12) for left, right in mapping.items())

checks["interval_matrix_complete"] = len(intervals) == 3 * 158
checks["stored_coverage_matches_margin"] = bool(((intervals.margin_A >= -1e-10) == intervals.covered_A).all() and ((intervals.margin_B >= -1e-10) == intervals.covered_B).all())
checks["margin_shapley_identity"] = float((intervals.delta_margin - intervals[["target_component", "centre_component", "radius_component"]].sum(axis=1)).abs().max()) < 1e-9
snl = transitions[transitions.dataset_group.eq("SNL")]
checks["snl_one_cell_transition_per_seed"] = all(int(snl[(snl.seed.eq(seed)) & snl.transition.eq("B_only")].n_cells.sum()) == 1 for seed in (42, 2021, 2024)) and not (snl.transition == "A_only").any()
checks["g34c2_is_training_outlier"] = len(labels[labels.file_name.eq("ISU-ILCC_G34C2.pkl") & labels.split.eq("train") & np.isclose(labels.label_delta_cycles, -2792)]) == 1
checks["g34c2_perturbation_complete"] = len(leverage) == 2 * 3 * 158

test = identity[identity.split.eq("test")].iloc[0]
checks["isu_identifiability_split_untouched"] = int(test.n_cells) == 48 and int(test.n_changed) == 27
checks["isu_identifiability_finite"] = all(np.isfinite(test[column]) for column in ("auroc", "average_precision", "brier_score", "auroc_bootstrap_95_low", "auroc_bootstrap_95_high"))
checks["early_feature_screen_excludes_metadata"] = len(manifest["parameters"]["identifiability_features"]) == 126 and not ({"split", "dataset_group", "chemistry_system", "file_name"} & set(manifest["parameters"]["identifiability_features"]))
checks["group_permutation_design_complete"] = bool(len(importance) == 14 and set(importance.split) == {"calibration", "test"} and importance.feature_group.nunique() == 7 and (importance.permutations == 1000).all())
checks["group_importance_reuses_fixed_scores"] = bool(all(np.isclose(importance.loc[importance.split.eq(split), "baseline_auroc"], identity.loc[identity.split.eq(split), "auroc"].iloc[0]).all() for split in ("calibration", "test")))
checks["magnitude_correlations_finite"] = bool(len(correlations) == 4 and np.isfinite(correlations.spearman_probability_vs_absolute_delta).all())
checks["auxiliary_output_hashes_match"] = all(sha256(aux / name) == digest for name, digest in aux_manifest["outputs_sha256"].items())
checks["input_hashes_match"] = all(sha256(path) == digest for path, digest in [(args.labels, manifest["inputs_sha256"]["labels"]), (args.errors, manifest["inputs_sha256"]["errors"]), (args.intervals, manifest["inputs_sha256"]["intervals"])])
checks["output_hashes_match"] = all(sha256(CORE / name) == digest for name, digest in manifest["outputs_sha256"].items())

result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "passed": sum(checks.values()), "total": len(checks)}
RESULT.parent.mkdir(parents=True, exist_ok=True)
RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
