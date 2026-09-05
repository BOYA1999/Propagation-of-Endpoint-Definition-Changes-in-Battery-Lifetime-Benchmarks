# Propagation of Endpoint-Definition Changes in Battery-Lifetime Benchmarks

This repository contains the code and derived inputs for a paired audit of battery-lifetime endpoint definitions. It compares an archived BatteryLife label vector (Contract A) with labels from a pinned generator (Contract B), holding the 823-cell cohort and its 503/162/158 training, calibration, and test split fixed.

The v14 analysis uses the controlled experiments stored under `reviewer_v13_20260904`, together with the v14 metadata ablation and endpoint diagnostics. Dates and version numbers in paths identify analysis stages; the original paths are retained so that results can be traced to their inputs.

## Reproduce the analysis

The companion input archive is `derived_data_v14.zip`. Upload of this archive to the repository's [Releases page](https://github.com/BOYA1999/Propagation-of-Endpoint-Definition-Changes-in-Battery-Lifetime-Benchmarks/releases) is pending. A local copy is required to run the analysis. Keep the ZIP outside the checkout and extract its enclosing `derived_data_v14/` directory into the repository root.

Run these commands from the repository root:

```text
python -m pip install -r requirements.txt
python scripts/verify_public_repo.py
python scripts/reproduce.py --data-root derived_data_v14 --output results
```

The reproduction command checks the code and input manifests, recalculates the main and metadata-ablation analyses from stored predictions using 5,000 paired seed/source-cell bootstrap replicates, runs independent numerical and rank checks, checks the stored ISU coordinate and Tongji traces, and generates five figures. Outputs are written to `results/`. `verification.json` records the numerical checks, and `reproduction.json` records the completed run. The output directory must be new or empty.

This route has been numerically checked on Windows. It uses stored predictions from the completed experiments; the full model-training campaign was not repeated for this release. Training inputs and commands are described in [INPUTS.md](INPUTS.md).

## Study design and findings

The four models are Ridge, XGBoost, CPMLP, and CPTransformer. The experiment uses 23 seed indices: 0–19, 42, 2021, and 2024. The main archive contains 184 completed model/contract/seed records and 58,880 calibration and test predictions. Ridge is deterministic; its repeated seed indices do not represent independent stochastic fits.

XGBoost ranks first in all 230 model-selection opportunities across two matched contracts, five error metrics, and 23 seeds. The complete four-model order agrees between contracts in 87 of 115 comparisons. Removing explicit source and chemistry indicators preserves the winner, while complete-order agreement is 80 of 115. That ablation uses 126 numerical features, 46 XGBoost fits, and two deterministic Ridge fits whose predictions are copied across seed indices; the deep-model predictions are reused.

For XGBoost, the ratio of cross-contract prediction movement to within-contract refit variation is 1.114 (95% bootstrap interval 0.984–1.261) over all 158 test cells. It is 1.534 (1.202–1.915) for the 44 cells with changed labels and 0.937 (0.832–1.060) for the 114 cells with unchanged labels. These comparisons support a subgroup-specific finding; they do not establish a uniform effect across the test cohort.

Both contracts express lifetime as a one-based position in the retained ordinary-cycle sequence. The ISU author endpoint, measured in calendar weeks, is a separate lineage diagnostic. Contract A has an exact post hoc effective reconstruction for 240 of 240 ISU cells, but its historical generator and execution record remain unavailable. Contract B uses BatteryLife commit `3b17bb53202e10711deae0448c91f84144bb3dd9`.

## Repository contents

| Location | Contents |
|---|---|
| `scripts/` | Reproduction entry point and release checks |
| `src/` | Notices directing users of earlier commands to `scripts/reproduce.py` |
| `02_experiments/` | Feature extraction, controlled training, analysis, and diagnostics |
| `06_figures/` | Figure construction scripts and plotting style |
| `derived_data_v14/` | Inputs extracted from the separate release archive |
| `results/` | Locally generated analysis outputs |

[INPUTS.md](INPUTS.md) describes file schemas and training preparation. [DATA_SOURCES.md](DATA_SOURCES.md) records source versions and attribution. [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) describes distribution, and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) records applicable third-party terms. Study code is covered by [LICENSE](LICENSE); source-derived data retain their upstream terms.
