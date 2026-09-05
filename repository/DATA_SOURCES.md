# Data sources

## Sources used in the analysis

| Asset | Version used | Purpose and source terms |
|---|---|---|
| [BatteryLife source code](https://github.com/Ruifeng-Tan/BatteryLife) | Commit `3b17bb53202e10711deae0448c91f84144bb3dd9` | Endpoint generator, processed-object conventions, and model implementations; MIT |
| [BatteryLife processed data](https://doi.org/10.5281/zenodo.21149533) | Zenodo record 21149533, v12 | Processed cell objects and archived lifetime labels; record metadata specifies CC BY 4.0 |
| [ISU-ILCC battery-aging data](https://doi.org/10.25380/iastate.22582234.v2) | Figshare v2 | Independent reconstruction of the author calendar-week endpoint; CC BY 4.0 |
| [ISU-ILCC feature-extraction code](https://doi.org/10.5281/zenodo.10714284) | Audited v1.1 archive; MD5 `c1fb05e9193046216c002770da6b029a` | Identification of the RPT/week endpoint; record metadata specifies CC BY 4.0, while the bundled license is CC0 1.0 |

The benchmark is described in the [BatteryLife paper](https://doi.org/10.1145/3711896.3737372). The [ISU-ILCC paper](https://doi.org/10.1016/j.xcrp.2024.101891) cites code DOI [10.5281/zenodo.10648587](https://doi.org/10.5281/zenodo.10648587); the exact v1.1 archive used here is record 10714284.

Contract A is the archived release-label vector. For ISU-ILCC, an effective rule reproduces 240 of 240 archived labels post hoc; the historical generator, input snapshot, and execution record have not been recovered. Contract B is reconstructed using the pinned BatteryLife implementation. Both labels use one-based positions in retained ordinary-cycle sequences. The ISU author endpoint is measured in calendar weeks and is used separately to examine source lineage.

## Sources represented in the cohort

The experiment uses BatteryLife's processed objects. The following landing pages document the original sources reported by the pinned BatteryLife repository.

| Source | Train / calibration / test | Original source |
|---|---:|---|
| CALCE | 9 / 2 / 2 | [CALCE Battery Data](https://calce.umd.edu/battery-data) |
| HNEI | 9 / 3 / 2 | [BatteryArchive](https://www.batteryarchive.org/index.html) |
| HUST | 47 / 15 / 15 | [Mendeley Data v2](https://data.mendeley.com/datasets/nsc7hnsg4s/2) |
| ISU-ILCC | 143 / 48 / 48 | [Iowa State Figshare v2](https://doi.org/10.25380/iastate.22582234.v2) |
| MATR | 102 / 34 / 33 | [MATR three-batch project](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204) and [batch 9](https://data.matr.io/1/projects/5d80e633f405260001c0b60a/batches/5dcef1fe110002c7215b2c94) |
| MICH | 24 / 8 / 8 | [BatteryArchive](https://www.batteryarchive.org/index.html) |
| MICH_EXP | 5 / 2 / 1 | [BatteryArchive](https://www.batteryarchive.org/index.html) |
| RWTH | 30 / 9 / 9 | [RWTH Aachen data archive](https://publications.rwth-aachen.de/record/818642/files/Rawdata.zip) |
| SNL | 30 / 10 / 10 | [BatteryArchive](https://www.batteryarchive.org/index.html) |
| Stanford | 25 / 8 / 8 | [Stanford formation dataset](https://data.matr.io/8/) |
| Tongji | 62 / 19 / 18 | [Zenodo record 6405084](https://zenodo.org/records/6405084) |
| UL_PUR | 2 / 0 / 0 | [BatteryArchive](https://www.batteryarchive.org/index.html) |
| XJTU | 15 / 4 / 4 | [Zenodo record 10963339](https://zenodo.org/records/10963339) |
| Total | 503 / 162 / 158 | 823 cells |

BatteryLife requests BatteryML citations for its processed CALCE, MATR, HUST, HNEI, RWTH, SNL, and UL_PUR subsets, and BatteryArchive citations for HNEI, SNL, MICH, MICH_EXP, and UL_PUR. Retain these source acknowledgments when reusing the data. The original providers' terms continue to apply.
