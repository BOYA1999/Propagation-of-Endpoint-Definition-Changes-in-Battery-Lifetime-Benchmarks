# Inputs and training

## Analysis inputs

Keep `derived_data_v14.zip` outside the checkout and extract its enclosing `derived_data_v14/` directory into the repository root. Keep the relative paths below intact and pass that directory to `scripts/reproduce.py --data-root`.

| Path under `derived_data_v14/` | Role |
|---|---|
| `03_data_interaction/data_integrity/paired_label_contracts/paired_label_contracts.csv` | Cell identifiers, fixed split, paired endpoint labels, and label-change status |
| `03_data_interaction/model_input/gate1_v2_20260821_0656/cycle100_features.csv` | Features calculated from the first 100 retained cycles |
| `03_data_interaction/model_input/gate1_v2_20260821_0656/labels/` | Separate training, calibration, and test labels for each contract |
| `03_data_interaction/taxonomy/cell_taxonomy.csv` | Source and chemistry descriptors and inclusion status |
| `04_analysis/reviewer_v13_20260904/core/` | Main stored predictions, training records, and frozen configuration |
| `04_analysis/reviewer_v14_20260905/metadata_ablation/` | Ablation predictions, training records, configuration, and comparison tables |
| `03_data_interaction/data_integrity/isu_contract_coordinate_audit/` | ISU coordinate reconstruction tables |
| `04_analysis/reviewer_v14_20260905/tongji_trace/` | Tongji endpoint trace and audit record |
| `06_figures/reviewer_v14_sync/figure_data/` | Archived figure source tables |

The archive also contains supporting endpoint and analysis tables at their original relative paths. These files retain the dates and version labels of the analysis stage that produced them. The v14 release uses the v13 controlled experiments as its main prediction input.

`MANIFEST.json` in the repository root and in the data root inventories the files distributed in each component and their SHA-256 hashes. The reproduction command checks both manifests. Frozen training configurations retain the provenance of the original experiments; their recorded source hashes describe the implementations used to generate the stored predictions, while the release manifests describe the public files.

The cell identifier is `file_name`. The paired-label table contains `split`, `dataset_group`, `release_eol`, `current_generator_eol`, and `labels_differ`. Prediction rows are identified by `fit_contract`, `split`, `file_name`, `model`, and `seed`; the prediction is `predicted_eol`. Contract names in files are `release` for A and `current_generator` for B. The supplied taxonomy contains `split`, `dataset_group`, `file_name`, `status`, `eol_cycle`, and `chemistry_system`; feature extraction selects its 823 rows with `status=usable`.

There are 823 included cells: 503 training, 162 calibration, and 158 test. Each cell has the same split under both contracts. Files named `test_labels_sealed.csv` preserve the original training-stage naming; they are supplied for public evaluation. Training reads the training-label files.

## Optional model training

Install the training dependencies in addition to the analysis environment:

```text
python -m pip install -r requirements-training.txt
```

The feature extractor requires trusted processed BatteryLife cell objects from [Zenodo record 21149533](https://doi.org/10.5281/zenodo.21149533). Arrange the extracted objects as `processed/<dataset_group>/<file_name>`, using the identifiers in the taxonomy table. These are Python pickle files and should be obtained from the identified source. Source archives and their terms are listed in [DATA_SOURCES.md](DATA_SOURCES.md).

Obtain a clean checkout of [BatteryLife](https://github.com/Ruifeng-Tan/BatteryLife) at commit `3b17bb53202e10711deae0448c91f84144bb3dd9`. The controlled training script checks that revision for every run and imports its implementations when training CPMLP or CPTransformer.

The examples below assume that the processed objects and BatteryLife checkout are in sibling directories `../processed` and `../BatteryLife`. They write new training artifacts outside the verified repository. Adjust those paths to match the downloaded files.

```text
python 02_experiments/baseline/extract_cycle_features.py --taxonomy derived_data_v14/03_data_interaction/taxonomy/cell_taxonomy.csv --processed ../processed --paired-labels derived_data_v14/03_data_interaction/data_integrity/paired_label_contracts/paired_label_contracts.csv --features-out ../training_inputs_v14/cycle100_features.csv --sequences-out ../training_inputs_v14/cycle100_sequences.npz --labels-root ../training_inputs_v14/labels --manifest-out ../training_inputs_v14/input_manifest.json
```

Feature extraction produces a CSV of numerical features and metadata, a `cycle100_sequences.npz` tensor, split-specific labels, and an input manifest. The tensor has shape `(823, 100, 3, 300)` and stores the first 100 retained cycles. Its arrays are `curves`, `split`, `dataset_group`, `chemistry_system`, and `file_name`.

The controlled training script accepts explicit feature, tensor, training-label, upstream-repository, and output paths. Its default model set is Ridge, XGBoost, CPMLP, and CPTransformer, with seed indices 0–19, 42, 2021, and 2024. The two contracts share the cohort, features, split, and fixed configurations. Deep-model pairs also share initialization, batch ordering, target scaling, and training budget.

```text
python 02_experiments/analysis/run_controlled_refits_v13.py --features ../training_inputs_v14/cycle100_features.csv --sequences ../training_inputs_v14/cycle100_sequences.npz --release-train-labels ../training_inputs_v14/labels/release/train_labels.csv --current-train-labels ../training_inputs_v14/labels/current_generator/train_labels.csv --repo ../BatteryLife --output ../training_v14 --device auto
```

The metadata ablation refits Ridge and XGBoost using the 126 numerical features after removing `dataset_group` and `chemistry_system` from their inputs. It performs 48 actual fits: 46 XGBoost fits and two deterministic Ridge fits. Ridge predictions are copied to the 23 seed indices, and the existing deep-model predictions are used in the four-model comparisons.

```text
python 02_experiments/analysis/run_metadata_ablation_v14.py fit --data-root derived_data_v14 --output ../metadata_fits_v14
python 02_experiments/analysis/run_metadata_ablation_v14.py analyze --data-root derived_data_v14 --fit-results ../metadata_fits_v14 --output ../metadata_analysis_v14
```

The default reproduction command evaluates stored predictions. For the public package, all 48 metadata-ablation fits were rerun and their 29,440 predictions matched the stored values exactly. A two-contract Ridge check also matched all 640 predictions. Deep-model checks covered forward execution; the full training campaign was not rerun.
