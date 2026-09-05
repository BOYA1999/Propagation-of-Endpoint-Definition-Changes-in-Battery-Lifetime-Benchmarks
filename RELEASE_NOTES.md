# v14 reproducibility files

Code and derived data for the paired endpoint-definition analysis in battery-lifetime benchmarks.

The data archive contains paired labels, early-cycle features, stored predictions, training records, endpoint traces and reference tables. The code recomputes the main and metadata-ablation results, including 5,000 paired seed/source-cell bootstrap replicates, model rankings, conformal summaries and five figures.

Extract `GitHub_Code_v14.zip`, then extract the enclosing `derived_data_v14/` directory from the data archive into the code repository. From the repository root:

```text
python -m pip install -r requirements.txt
python scripts/reproduce.py --data-root derived_data_v14 --output results
```

The archive replay passed all 63 numerical checks in a fresh Windows Python 3.12 environment. The 48 metadata-ablation fits were also rerun and reproduced the stored predictions exactly. The full deep-model training campaign was not repeated. Raw processed battery objects and model weights are not included; source links and optional training commands are documented in the repository.

Contract A remains an archived label vector with a post hoc effective ISU reconstruction, not a recovered historical generator. Both model targets use retained ordinary-cycle positions; the separate author calendar-week endpoint is a lineage diagnostic.

SHA-256:

```text
1e193bc255e7ae4724a9fbdccac250b980a538219e09dc3d4582345243ed494c  GitHub_Code_v14.zip
2e088e1c0d782e066fc91df1cdea791ba3fea40424ad8c0f539c7c24c9cf31c2  derived_data_v14.zip
```
