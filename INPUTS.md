# Input contract

All input paths are supplied explicitly. The repository does not include these files.

## Endpoint regeneration

`audit_endpoint_labels.py` requires:

- `--taxonomy`: CSV with `split`, `dataset_group`, and `file_name`;
- `--processed`: directory containing the separately obtained processed cell objects;
- `--repo`: BatteryLife checkout fixed to commit `3b17bb53202e10711deae0448c91f84144bb3dd9`;
- `--out`: new or empty output directory.

## Core paired audit

`analyze_endpoint_contracts.py` requires:

- `paired_label_contracts.csv`: `split`, `dataset_group`, `file_name`, `release_eol`, `current_generator_eol`, `label_relation`, `label_delta_cycles`, `labels_differ`;
- `endpoint_label_cells.csv`: output of `audit_endpoint_labels.py`;
- `predictions/release/revealed_predictions.csv` and `predictions/current_generator/revealed_predictions.csv`;
- `gate4_metrics.csv`, containing the prespecified interval robustness summaries.

`build_tables.py` additionally reads `cohort_validity.csv`, `source_coverage_matrix.csv`, and `two_by_two_test_metrics.csv` from the corresponding analysis directories.

## Deeper mechanism analysis

`run_depth_campaign.py` expects the following six tables:

| Argument | Required content |
|---|---|
| `--labels` | the eight columns listed above for `paired_label_contracts.csv` |
| `--branches` | `file_name` and `endpoint_kind`, with one branch record per cell |
| `--errors` | `dataset_group`, `file_name`, `model`, `seed`, `predicted_eol`, `fit_contract`, `evaluation_contract`, `evaluation_eol`, `absolute_percentage_error` |
| `--intervals` | matched-contract interval predictions including `contract`, `model`, `seed`, `nominal`, `method`, `file_name`, `dataset_group`, `eol_cycle`, `predicted_eol`, `calibration_quantile`, and `covered` |
| `--gate4` | perturbation predictions including `analysis`, `contract`, `seed`, `nominal`, `file_name`, `dataset_group`, `predicted_eol`, `calibration_quantile`, and `covered` |
| `--features` | `split`, `dataset_group`, `chemistry_system`, `file_name`, followed by the 126 cycle-100 numerical features |

The feature groups are charge capacity, discharge capacity, coulombic efficiency, charge voltage, discharge voltage, charge C-rate, and discharge C-rate. Metadata columns are excluded from the identifiability model.

Validation command:

```powershell
python src/validate_depth_campaign.py `
  --labels inputs/paired_label_contracts.csv `
  --core outputs/depth/core `
  --published-effects inputs/two_by_two_factor_effects_by_seed.csv `
  --errors inputs/two_by_two_cellwise_errors.csv `
  --intervals inputs/interval_predictions.csv `
  --auxiliary outputs/depth/auxiliary `
  --output outputs/depth/depth_validation.json
```

## Figure source tables

Place these eight CSV files in the directory passed to `build_paper_figures.py --data`:

| File | Required columns |
|---|---|
| `Fig2_target_geometry_cells.csv` | paired labels plus `endpoint_kind`, `mean_eol`, `absolute_delta_cycles`, `relative_delta`, `absolute_relative_delta` |
| `Fig3_status.csv` | model, seed, label status, counts, within-group effects, full-mean contributions, median absolute prediction shift |
| `Fig3_bootstrap.csv` | model, metric, bootstrap median, 95% limits, replicate count |
| `Fig3_shifts.csv` | model, seed, unchanged-label share and changed/unchanged prediction-shift summaries |
| `Fig4_metric_factor_summary.csv` | model, metric, evaluation and fitting medians/ranges, interaction median |
| `Fig5_interval_transitions.csv` | seed, source, transition, cell count |
| `Fig5_margin_components.csv` | seed, source, test count, coverage counts, target/centre/radius margin components, coverage step |
| `Fig5_g34c2_leverage.csv` | contract, seed, source, cell count, prediction and quantile shifts, coverage before/after removal |

The figure builder checks filenames by reading them directly. Missing or altered schemas fail at the point of use rather than being silently inferred.

