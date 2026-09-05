# Third-party notices

The repository's [MIT license](LICENSE) covers the study code and documentation. Source-derived labels, descriptors, and measurements retain the attribution and terms of their original providers. Study-generated predictions and analysis outputs are supplied for reproduction under the repository terms; these terms do not override upstream rights.

## BatteryLife

This study uses [BatteryLife](https://github.com/Ruifeng-Tan/BatteryLife) at commit `3b17bb53202e10711deae0448c91f84144bb3dd9`. Feature resampling follows its data-loader logic, and optional deep-model training imports its CPMLP and CPTransformer implementations. The upstream MIT notice is reproduced below.

> MIT License
>
> Copyright (c) 2025 Ruifeng Tan（谭瑞锋）
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

BatteryLife's processed-data release used here is [Zenodo record 21149533](https://doi.org/10.5281/zenodo.21149533), version 12. Its metadata specifies CC BY 4.0. Derived tables identify the source cells and retain their source lineage.

## ISU-ILCC

The endpoint-semantic audit uses [ISU-ILCC data, Figshare v2](https://doi.org/10.25380/iastate.22582234.v2), under CC BY 4.0, and the [v1.1 feature-extraction code archive](https://doi.org/10.5281/zenodo.10714284). The audited code archive has MD5 `c1fb05e9193046216c002770da6b029a`. Its Zenodo metadata specifies CC BY 4.0, while its bundled `LICENSE` contains CC0 1.0. Both notices are retained here. The original paper cites the separate code DOI [10.5281/zenodo.10648587](https://doi.org/10.5281/zenodo.10648587).

## Other source datasets and dependencies

The cohort includes records originating from CALCE, HNEI, HUST, ISU-ILCC, MATR, MICH, MICH_EXP, RWTH, SNL, Stanford, Tongji, UL_PUR, and XJTU. [DATA_SOURCES.md](DATA_SOURCES.md) lists their landing pages and the attribution requested by BatteryLife. Raw archives and processed cell objects are obtained from the source providers under their terms.

Software installed through the requirements files is governed by each dependency's license.
