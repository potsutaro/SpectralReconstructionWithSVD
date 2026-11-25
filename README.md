# Spectral Reconstruction with Singular Value Decomposition

A Python framework for solving inverse problems in lattice QCD using singular value decomposition (SVD), based on [arXiv:2605.15674](https://arxiv.org/abs/2605.15674).

The framework provides a workflow from Euclidean correlation functions to smeared spectral functions, high-pass-filtered correlators, and energy moments. It also supports binned bootstrap resampling and mock-data studies.

---

# Method

The Euclidean correlator is related to the spectral function through the Laplace transform:

$$
C(t)=\int_0^\infty d\omega\,e^{-\omega t}\rho(\omega).
$$

Reconstructing the spectral function from a finite set of noisy correlator data is an ill-conditioned inverse problem. After discretization, the kernel can be decomposed as

$$
K=U\Sigma V^\dagger.
$$

The correlator is projected onto the singular modes, and modes supported by the statistical precision of the data can be retained to construct a regularized spectral reconstruction. The truncation limits the attainable spectral resolution. See the reference paper for the detailed formalism and treatment of uncertainties.

---

# Workflow

```text
       Raw correlators           Mock correlators
              |                   gen_mock_Ct.py
              |                         |
              +------------+------------+
                           |
                           v
             +---------------------------+
             | Binned bootstrap sampling |
             | data_binnedbootstrap.py   |
             +-------------+-------------+
                           |
                           v
             +---------------------------+
             | Correlator preprocessing  |
             | sweep_subtraction.py      |
             +-------------+-------------+
                           |
                           v
             +----------------------------------+
             | SVD modal decomposition          |
             | sweep_BStrapModalDecomposition.py|
             +----------------+-----------------+
                              |
               +--------------+--------------+
               |              |              |
               v              v              v
         Smeared SPF    High-pass filter  Energy moments
         (SmearedSPF)    (HPassFilter)    (EnergyMoment)
```

The `data` and `mock` tasks perform the preprocessing and modal decomposition. The three subsequent analysis tasks can be executed separately using the prepared data.

---

# Directory Structure

```text
project/
├── run.py                              # Main entry point
├── config.py                           # Configuration classes
├── clean.sh                            # Cleanup script
├── configs/
│   ├── default.yaml
│   └── mock.yaml
├── analysis/
│   ├── DataNormalization.py
│   ├── GenMockData.py
│   ├── SmearedSPF.py
│   ├── data_bootstrap.py
│   ├── data_binnedbootstrap.py
│   ├── gen_mock_Ct.py
│   ├── sweep_subtraction.py
│   ├── sweep_BStrapModalDecomposition.py
│   ├── sweep_BStrapSmearedSPF.py
│   ├── sweep_BStrapHFilteringCt.py
│   └── sweep_BStrapEnergyMoment.py
├── raw_data/                           # Raw input data
├── correlator_data/                    # Correlator datasets
├── subtracted_data/                    # Preprocessed correlators
├── modaldecomposition_data/            # Modal decomposition data
├── singularvalues_data/                # Singular-value data
├── smearedSPF_data/                     # Smeared spectral-function data
├── smearingfactor_data/                 # Smearing-factor data
└── results/
    ├── EnergyMoment/
    ├── SpecFunc/
    └── mass/
```

---

# Installation

Clone the repository and move to its root directory:

```bash
git clone <repository-url>
cd <repository-directory>
```

Install the expected Python dependencies:

```bash
python -m pip install numpy scipy matplotlib pyyaml
```

---

# Configuration

Analysis parameters are specified in a YAML file and loaded through `Config.from_yaml()`.

Example:

```yaml
lattice:
  T: 96
statistics:
  stati: 1  # 1 for bootstrap
  configuration: 100
  bootstrap: 500
  binsize: 1
analysis:
  normalization: 1
  init: 0
  size: 21
  dof: 22
  fitini: 12
  fitfin: 16
  delta: 0.1
  wmax: 0.01
  wmin: 3.01
  nw: 1000
  nin: 0
  ntr: 7
  neb: 6
mock:
  state: 2
  error: 0.01
paths:
  corr: correlator_data
files:
  rdata: raw_data/cdata_jk
  cdata: cdata
  ndata: ndata
```

### `lattice`

| Parameter | Description |
| --- | --- |
| `T` | Temporal lattice extent. |

### `statistics`

| Parameter | Description |
| --- | --- |
| `stati` | Statistical analysis mode (`1` = bootstrap). |
| `configuration` | Number of input gauge configurations or mock samples. |
| `bootstrap` | Number of bootstrap samples. |
| `binsize` | Bin size used for bootstrap resampling. |

### `analysis`

| Parameter | Description |
| --- | --- |
| `normalization` | Time slice used for correlator normalization. |
| `init` | Initial time slice of the analysis. |
| `size` | Final time slice of the analysis. |
| `dof` | Number of correlator data points used in the analysis (22 for the example). |
| `fitini` | First time slice included in the fit. |
| `fitfin` | Last time slice included in the fit. |
| `delta` | Smearing-width parameter. |
| `wmax` | Minimum energy of the spectral reconstruction grid. |
| `wmin` | Maximum energy of the spectral reconstruction grid. |
| `nw` | Number of energy grid points used for spectral reconstruction. |
| `nin` | First mode included in the analysis (fixed to 0). |
| `ntr` | SVD truncation parameter. |
| `neb` | Last mode included in the determination of the error band. |

### `mock`

| Parameter | Description |
| --- | --- |
| `state` | Number of states used to generate mock correlators. |
| `error` | Noise parameter for mock correlators. |

### `paths`

| Parameter | Description |
| --- | --- |
| `corr` | Directory containing correlator data. |

### `files`

| Parameter | Description |
| --- | --- |
| `rdata` | Input raw correlator data path. |
| `cdata` | Name of the preprocessed correlator dataset. |
| `ndata` | Name of the normalized correlator dataset. |

---

# Running

General syntax (run from the repository root):

```bash
python run.py <config.yaml> --task <task>
```

## Analyze existing lattice data

```bash
python run.py configs/default.yaml --task data
```

## Generate and analyze mock data

```bash
python run.py configs/mock.yaml --task mock
```

## Reconstruct smeared spectral functions

```bash
python run.py configs/default.yaml --task SmearedSPF
```

## Perform high-pass filtering

```bash
python run.py configs/default.yaml --task HPassFilter
```

## Evaluate energy moments

```bash
python run.py configs/default.yaml --task EnergyMoment
```

For mock-data studies, use `configs/mock.yaml` in the subsequent analysis commands as well. Run the corresponding `data` or `mock` task before the downstream analysis tasks. Task names are case-sensitive.

---

# Available Tasks

| Task | Description | Analysis module(s) |
| --- | --- | --- |
| `data` | Binned bootstrap, correlator preprocessing, and SVD modal decomposition. | `data_binnedbootstrap.py`, `sweep_subtraction.py`, `sweep_BStrapModalDecomposition.py` |
| `mock` | Generate mock correlators and perform the same preprocessing and modal decomposition. | `gen_mock_Ct.py` followed by the modules used in `data` |
| `SmearedSPF` | Reconstruct smeared spectral functions. | `sweep_BStrapSmearedSPF.py` |
| `HPassFilter` | Perform high-pass filtering of correlators. | `sweep_BStrapHFilteringCt.py` |
| `EnergyMoment` | Evaluate energy moments. | `sweep_BStrapEnergyMoment.py` |

---

# Input and Output

The raw correlator input path is specified by `files.rdata`. The mock-data workflow provides an alternative to external lattice data for validation studies.

The repository contains directories for bootstrap and preprocessed correlators, singular values, modal decomposition, smeared spectral functions, and smearing factors.

Example result files from the supplied directory structure include:

| File | Observable |
| --- | --- |
| `results/SpecFunc/ReconstSSpecFDelta=0.3Ntr=7.pdf` | Smeared spectral reconstruction |
| `results/SpecFunc/ReconstSpecFNtr=7.pdf` | Spectral reconstruction |
| `results/EnergyMoment/ReconstEMom0thNtr=7.pdf` | Zeroth energy moment |
| `results/EnergyMoment/ReconstEMom1stNtr=7.pdf` | First energy moment |
| `results/EnergyMoment/ReconstEMom2ndNtr=7.pdf` | Second energy moment |

---

# Extending the Framework

Each analysis module should implement

```python
def main(cfg):
    ...
```

To add a new analysis:

1. Create a new module under `analysis/`.
2. Implement `main(cfg)`.
3. Import the module in `run.py`.
4. Add a new task in the task dispatcher.

---

# Citation

If this software contributes to published work, please cite:

R. Tsuji and S. Hashimoto, *Spectral reconstruction from Euclidean lattice correlators through singular value decomposition*, [arXiv:2605.15674](https://arxiv.org/abs/2605.15674) [hep-lat] (2026).

```bibtex
@misc{Tsuji:2026svd,
  author        = {Tsuji, Ryutaro and Hashimoto, Shoji},
  title         = {Spectral reconstruction from Euclidean lattice correlators through singular value decomposition},
  year          = {2026},
  eprint        = {2605.15674},
  archivePrefix = {arXiv},
  primaryClass  = {hep-lat},
  doi           = {10.48550/arXiv.2605.15674}
}
```
