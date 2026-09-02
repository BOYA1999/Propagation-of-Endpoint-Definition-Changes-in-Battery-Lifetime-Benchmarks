# Data sources and provenance

## Sources used directly

| Asset | Role in the study | Fixed version | Official source | Licence or terms |
|---|---|---|---|---|
| BatteryLife source code | endpoint generator and processed-object rules | commit `3b17bb53202e10711deae0448c91f84144bb3dd9` | [GitHub](https://github.com/Ruifeng-Tan/BatteryLife) | MIT |
| BatteryLife processed data | processed cell objects and archived life labels | v12, published 3 July 2026 | [Zenodo record 21149533](https://zenodo.org/records/21149533) | CC BY 4.0 in the record metadata |
| BatteryLife paper | benchmark description and citation | KDD 2025 | [DOI 10.1145/3711896.3737372](https://doi.org/10.1145/3711896.3737372) | publisher terms |
| ISU-ILCC battery-aging data | independent endpoint-semantic reconstruction | Figshare version 2 | [DOI 10.25380/iastate.22582234.v2](https://doi.org/10.25380/iastate.22582234.v2) | CC BY 4.0 |
| ISU-ILCC feature-extraction code | archived implementation used to identify the RPT/week endpoint | audited v1.1 archive, local MD5 `c1fb05e9193046216c002770da6b029a` | [Zenodo record 10714284](https://doi.org/10.5281/zenodo.10714284) | record metadata: CC BY 4.0; bundled `LICENSE`: CC0 1.0 |
| ISU-ILCC paper | experimental procedure and original code-availability statement | 2024 | [DOI 10.1016/j.xcrp.2024.101891](https://doi.org/10.1016/j.xcrp.2024.101891) | article terms |

The published ISU-ILCC paper points to Zenodo DOI `10.5281/zenodo.10648587` for its feature-extraction code. The audit used the separately pinned v1.1 archive at record `10714284`. Both identifiers are reported so that the paper-level citation and the exact audited snapshot are not conflated.

## Original source landing pages represented in the 823-cell cohort

The numerical analysis used the processed BatteryLife v12 objects, not a new merge of the raw archives below. These links document the original source lineage exposed by the pinned BatteryLife repository.

| Cohort source | Train / calibration / test | Original landing page reported by BatteryLife |
|---|---:|---|
| CALCE | 9 / 2 / 2 | [University of Maryland CALCE Battery Data](https://calce.umd.edu/battery-data) |
| HNEI | 9 / 3 / 2 | [BatteryArchive](https://www.batteryarchive.org/index.html) |
| HUST | 47 / 15 / 15 | [Mendeley Data, version 2](https://data.mendeley.com/datasets/nsc7hnsg4s/2) |
| ISU-ILCC | 143 / 48 / 48 | [Iowa State University Figshare](https://iastate.figshare.com/articles/dataset/_b_ISU-ILCC_Battery_Aging_Dataset_b_/22582234) |
| MATR | 102 / 34 / 33 | [MATR three-batch project](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204) and [batch 9](https://data.matr.io/1/projects/5d80e633f405260001c0b60a/batches/5dcef1fe110002c7215b2c94) |
| MICH | 24 / 8 / 8 | [BatteryArchive](https://www.batteryarchive.org/index.html) |
| MICH_EXP | 5 / 2 / 1 | [BatteryArchive](https://www.batteryarchive.org/index.html) |
| RWTH | 30 / 9 / 9 | [RWTH Aachen raw-data archive](https://publications.rwth-aachen.de/record/818642/files/Rawdata.zip) |
| SNL | 30 / 10 / 10 | [BatteryArchive](https://www.batteryarchive.org/index.html) |
| Stanford | 25 / 8 / 8 | [MATR Stanford formation dataset](https://data.matr.io/8/) |
| Tongji | 62 / 19 / 18 | [Zenodo record 6405084](https://zenodo.org/records/6405084) |
| UL_PUR | 2 / 0 / 0 | [BatteryArchive](https://www.batteryarchive.org/index.html) |
| XJTU | 15 / 4 / 4 | [Zenodo record 10963339](https://zenodo.org/records/10963339) |
| **Total** | **503 / 162 / 158** | **823 cells** |

BatteryLife also asks users of its processed CALCE, MATR, HUST, HNEI, RWTH, SNL, and UL_PUR subsets to cite BatteryML, and users of HNEI, SNL, MICH, MICH_EXP, and UL_PUR to cite BatteryArchive. Follow the current upstream citation instructions when preparing a publication.

## Inclusion and redistribution boundary

- None of the raw landing-page archives is included here.
- No BatteryLife processed object or life-label file is included here.
- No ISU-ILCC raw file or upstream source file is included here.
- Study-derived tables used by the scripts are distributed with the manuscript's executable online resource, not in this code-only package.
- Anyone downloading an upstream asset is responsible for following the terms shown by that provider at the time of access.

