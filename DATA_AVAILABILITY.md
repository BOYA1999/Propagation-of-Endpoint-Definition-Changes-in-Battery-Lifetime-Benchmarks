# Data availability

This repository includes `derived_data_v14.zip`, containing study-derived labels, cell descriptors, numerical features, stored predictions, experiment records, diagnostic tables and figure inputs. It does not include the upstream raw archives, processed cell objects or trained weights. The v20 manuscript revision introduced no new scientific data or analysis code; the existing archive remains the replay input.

## Obtaining the upstream material

1. Obtain the BatteryLife repository and check out commit `3b17bb53202e10711deae0448c91f84144bb3dd9`.
2. Download BatteryLife processed release v12 from Zenodo record `21149533` and follow the provider's licence and citation instructions.
3. For the independent ISU endpoint-semantic reconstruction, obtain the ISU-ILCC version 2 data from Iowa State University Figshare and the pinned feature-extraction code archive identified in [DATA_SOURCES.md](DATA_SOURCES.md).
4. For the full v14 replay, obtain the study-derived tables from `derived_data_v14.zip` and use the independently extracted `GitHub_Code_v14.zip` as described in [README.md](README.md). The root-level [INPUTS.md](INPUTS.md) describes the separate earlier code-only snapshot.

The code repository has no fabricated public data DOI. A persistent identifier for the final study-specific resource should be inserted only after the authors have deposited the approved release.

## What can be reproduced from each layer

| Layer | Required material | Result |
|---|---|---|
| Endpoint regeneration | pinned BatteryLife checkout, processed cell objects, taxonomy table | regenerated endpoint-label audit table |
| Core paired audit | paired labels, endpoint audit, revealed predictions, interval summary | cohort flow, endpoint differences, 2 x 2 effects, robustness summaries |
| Mechanism analysis | paired labels, branch records, cellwise point errors, interval predictions, early-cycle features | rank, concentration, propagation, metric, margin, leverage, and identifiability diagnostics |
| Figure production | eight derived figure-source tables | five editable and high-resolution figure sets |

The public derived archive contains battery-cell identifiers and numerical results. Source-derived data retain the relevant upstream attribution and terms; see [DATA_SOURCES.md](DATA_SOURCES.md) and [repository/THIRD_PARTY_NOTICES.md](repository/THIRD_PARTY_NOTICES.md). The archive supports replay from stored predictions; it does not constitute a fresh training run or validation of the v20 manuscript text.

