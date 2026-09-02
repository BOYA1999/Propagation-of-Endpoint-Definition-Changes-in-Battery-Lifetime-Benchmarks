# Data availability

This is a code-only repository. It intentionally excludes third-party data, study-derived input tables, raw archives, processed cell objects, trained weights, and prediction files.

## Obtaining the upstream material

1. Obtain the BatteryLife repository and check out commit `3b17bb53202e10711deae0448c91f84144bb3dd9`.
2. Download BatteryLife processed release v12 from Zenodo record `21149533` and follow the provider's licence and citation instructions.
3. For the independent ISU endpoint-semantic reconstruction, obtain the ISU-ILCC version 2 data from Iowa State University Figshare and the pinned feature-extraction code archive identified in [DATA_SOURCES.md](DATA_SOURCES.md).
4. Supply the study-derived tables listed in [INPUTS.md](INPUTS.md). During anonymous review, those tables and their hashes accompany the manuscript in its executable online resource.

The code repository has no fabricated public data DOI. A persistent identifier for the final study-specific resource should be inserted only after the authors have deposited the approved release.

## What can be reproduced from each layer

| Layer | Required material | Result |
|---|---|---|
| Endpoint regeneration | pinned BatteryLife checkout, processed cell objects, taxonomy table | regenerated endpoint-label audit table |
| Core paired audit | paired labels, endpoint audit, revealed predictions, interval summary | cohort flow, endpoint differences, 2 x 2 effects, robustness summaries |
| Mechanism analysis | paired labels, branch records, cellwise point errors, interval predictions, early-cycle features | rank, concentration, propagation, metric, margin, leverage, and identifiability diagnostics |
| Figure production | eight derived figure-source tables | five editable and high-resolution figure sets |

The study-specific derived inputs contain battery-cell identifiers and numerical results, not personal or clinical information. They are nevertheless kept outside this code-only repository so that data licensing and archival citation remain explicit.

