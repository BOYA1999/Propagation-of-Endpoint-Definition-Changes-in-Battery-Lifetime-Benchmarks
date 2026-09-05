# Data availability

The analysis uses BatteryLife processed data from [Zenodo record 21149533](https://doi.org/10.5281/zenodo.21149533), version 12, and the endpoint-generation code at BatteryLife commit `3b17bb53202e10711deae0448c91f84144bb3dd9`. The Zenodo record identifies the processed data as CC BY 4.0. Original sources and the separate ISU endpoint audit materials are listed in [DATA_SOURCES.md](DATA_SOURCES.md).

The companion archive `derived_data_v14.zip` contains paired labels, cell descriptors, numerical features, stored predictions, experiment records, diagnostic tables, and figure source data. Its enclosing directory is `derived_data_v14/`, with source-relative paths retained. The archive has been prepared for manual upload to the repository's [Releases page](https://github.com/BOYA1999/Propagation-of-Endpoint-Definition-Changes-in-Battery-Lifetime-Benchmarks/releases); that upload is pending.

These inputs support recalculation of the paired endpoint analyses, 5,000-replicate bootstrap summaries, metadata ablation, rank and endpoint diagnostics, and five figures. They also preserve the training configurations and prediction records needed to inspect the experimental comparisons. Instructions are in [README.md](README.md) and [INPUTS.md](INPUTS.md).

Raw source archives, processed cell objects, trained weights, and manuscript files are not distributed in this package. Processed objects for feature extraction and model training are obtained from the upstream data repository. The historical generator for Contract A is unavailable; the release supplies the archived labels and the evidence for its post hoc effective reconstruction.

Source-derived labels, descriptors, and measurements retain their upstream attribution and applicable terms. Study-generated predictions and analysis outputs are released for reproduction under the repository terms, which do not override the terms of their source materials. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
