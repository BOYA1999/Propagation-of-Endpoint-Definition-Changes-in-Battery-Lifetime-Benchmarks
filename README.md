# Endpoint-definition audit for BatteryLife

This repository contains the study-specific analysis code for examining how an executable end-of-life definition propagates through a fixed battery-lifetime benchmark. It compares archived BatteryLife release labels with labels regenerated from a pinned source revision while holding cell identities, early-cycle inputs, data splits, models, and random seeds fixed.

The repository contains two public v14 reproduction archives: `GitHub_Code_v14.zip` and `derived_data_v14.zip`. The data archive includes study-derived labels, cell descriptors, numerical features, stored predictions, experiment records, and diagnostics. Raw processed cell objects and trained model weights are not included. Source-derived material remains subject to its upstream licence and attribution requirements.

## v20 manuscript alignment and archived reproduction

The v20 manuscript revision changes the writing and reference numbering; it does not introduce new model fits, scientific datasets, numerical results, or analysis code. The existing v14 code and derived-data archives remain the relevant stored-prediction replay. `reference_number_mapping_v19_v20.csv` records the manuscript reference-number changes; it is bibliographic metadata, not a new experimental dataset. The manuscript and author title page are not distributed here.

For the full v14 replay, download both archives from this repository, extract `GitHub_Code_v14.zip` to a new directory, and extract the enclosing `derived_data_v14/` directory into its extracted `repository/` directory. Run `python -m pip install -r requirements.txt`, `python scripts/verify_public_repo.py --data-root derived_data_v14`, and `python scripts/reproduce.py --data-root derived_data_v14 --output results` from that extracted `repository/` directory. The checked-in `repository/` directory is the historical code snapshot; its statements that the data archive is pending release predate the archive now available at the repository root. The root-level `src/` and `scripts/` below are a separate, earlier code-only snapshot, not the full replay entry point.

The root `MANIFEST.sha256` lists files in the earlier code-only snapshot and is not a manifest for the full current GitHub repository. The archive checksums are in `SHA256SUMS.txt`; the code and data archives contain their own inventory manifests. Historical replay and validation records do not validate the v20 manuscript text or constitute new model training.

## Scientific scope

The audited cohort contains 823 Li-ion cells from 13 sources, divided into 503 training, 162 calibration, and 158 test cells. The two endpoint contracts assign different labels to 252 cells. The analysis treats both as versioned computational contracts; it does not designate either contract as physical ground truth.

The code supports four linked analyses:

1. reconstruction of generator-side endpoint labels from separately obtained BatteryLife objects;
2. paired endpoint-contract and 2 x 2 fit/evaluate analyses;
3. source, rank, propagation, metric, interval-margin, and identifiability diagnostics;
4. production and validation of the five figures used in the current manuscript.

## Repository layout

```text
src/
  audit_endpoint_labels.py
  analyze_endpoint_contracts.py
  validate_results.py
  build_tables.py
  run_depth_campaign.py
  run_identifiability_diagnostics.py
  validate_depth_campaign.py
  build_paper_figures.py
  validate_paper_figures.py
  academic.mplstyle
scripts/
  verify_public_repo.py
```

The first four scripts reconstruct the core audit and tables. The next three run and verify the deeper mechanism analyses. The last two build and inspect the current five-figure set.

## Environment

Python 3.12 was used for the archived run.

```powershell
python -m pip install -r requirements.txt
```

## Inputs

Inputs are supplied explicitly on the command line. No script assumes a personal directory or a particular repository checkout. See [INPUTS.md](INPUTS.md) for filenames, required columns, and their provenance.

The main external dependencies are:

- BatteryLife source code at commit `3b17bb53202e10711deae0448c91f84144bb3dd9`;
- BatteryLife processed release v12, Zenodo record `21149533`;
- the ISU-ILCC battery-aging data and the archived ISU feature-extraction code for the independent endpoint-semantic check.

Exact landing pages and licence boundaries are listed in [DATA_SOURCES.md](DATA_SOURCES.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Core audit

```powershell
python src/audit_endpoint_labels.py `
  --taxonomy inputs/taxonomy.csv `
  --processed inputs/processed_cells `
  --repo inputs/BatteryLife `
  --out outputs/endpoint_audit

python src/analyze_endpoint_contracts.py `
  --paired-labels inputs/paired_label_contracts.csv `
  --endpoint-audit outputs/endpoint_audit/endpoint_label_cells.csv `
  --prediction-root inputs/predictions `
  --gate4-metrics inputs/gate4_metrics.csv `
  --output outputs/core

python src/validate_results.py `
  --core outputs/core `
  --output outputs/core_validation.json
```

## Deeper analysis

```powershell
python src/run_depth_campaign.py `
  --labels inputs/paired_label_contracts.csv `
  --branches inputs/endpoint_generation_records.csv `
  --errors inputs/two_by_two_cellwise_errors.csv `
  --intervals inputs/interval_predictions.csv `
  --gate4 inputs/gate4_interval_predictions.csv `
  --features inputs/cycle100_features.csv `
  --output outputs/depth/core

python src/run_identifiability_diagnostics.py `
  --features inputs/cycle100_features.csv `
  --labels inputs/paired_label_contracts.csv `
  --output outputs/depth/auxiliary
```

`validate_depth_campaign.py` compares those outputs with the frozen paired labels, point-effect table, and interval inputs. Its full command is given in [INPUTS.md](INPUTS.md).

## Figures

The figure builder expects eight derived source tables. Their schemas are documented in [INPUTS.md](INPUTS.md).

```powershell
python src/build_paper_figures.py `
  --data inputs/figure_data `
  --output outputs/figures

python src/validate_paper_figures.py `
  --figures outputs/figures `
  --data inputs/figure_data `
  --output outputs/figure_validation.json
```

The builder exports editable SVG, PDF, EPS, and 600-dpi PNG files. Each figure is organized around one bounded conclusion; source colours, route colours, marker shapes, and panel labels are held consistent across the set.

## Earlier code-only snapshot check

The following command belongs to the earlier standalone code-only snapshot. It does not validate the current mixed repository, whose additional files are outside that snapshot’s allowlist:

```powershell
python scripts/verify_public_repo.py
```

The check verifies the allowlisted file set, Python syntax and command-line entry points, the MIT licence, the package hash manifest, and the absence of raw data, model artefacts, caches, local paths, email addresses, credentials, internal workflow labels, and automated-writing residue.

## Citation and identifiers

Please cite the BatteryLife paper and the original data sources listed in [DATA_SOURCES.md](DATA_SOURCES.md). A DOI for this code repository has not been invented or reserved in this package. Repository citation metadata should be added only after the final authorship and persistent identifier are available.

## Licence

The study-specific code in this repository is released under the [MIT License](LICENSE). That licence does not replace the licences or terms of the BatteryLife repository, BatteryLife data, ISU-ILCC data, or any other upstream resource.
