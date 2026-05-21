# 21cmFAST + patchy kSZ + non-Gaussianity through the EoR

A small pipeline that:

1. installs 21cmFAST + deps,
2. runs 21cmFAST coeval cubes and a lightcone,
3. computes the patchy kSZ signal via a simple coeval-cube method
   and via a self-contained lightcone routine (with / without rotation),
4. computes non-Gaussian statistics (PDF, variance, skewness,
   kurtosis, 1D power spectrum),
5. tracks those statistics as a function of mean neutral fraction
   x_HI across the EoR.

## Files

- `run_pipeline.ipynb` — top-to-bottom driver. Config, simulation
  runs, and non-Gaussian code are inline cells.
- `ksz.py` — kSZ functions (`ksz_from_coeval`, `ksz_from_lightcone`,
  `ksz_squared_from_lightcone`). The lightcone routines are
  self-contained: `run_kSZ`, `run_kSZ_sq`, `_Proj_array`,
  `_KszConstants`, and `KSZOutput` are all inlined here, so a stock
  `py21cmfast` install is enough.
- `plotting.py` — plot helpers (`plot_evolution`,
  `plot_pdf_evolution`, `plot_power_spectrum`, `plot_ksz_map`,
  `plot_lightcone_pdf`).
- `requirements.txt` — Python dependencies.

## Install

21cmFAST is a C + Python code and needs `gsl` and `fftw3` dev
libraries on the system.

**conda (recommended, works on HPC without sudo):**

```bash
module load Miniconda3          # or install Miniforge if no module
conda init bash                  # once, then reopen your shell
# (alternative: 'source $(conda info --base)/etc/profile.d/conda.sh' for just this shell)

conda create -n ksz21 -c conda-forge -y python=3.10 gsl fftw c-compiler pkg-config
conda activate ksz21
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name ksz21 --display-name "Python (ksz21)"
```


If `conda` itself is missing, install Miniforge into your home:

```bash
curl -fsSL "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -o mf.sh
bash mf.sh -b -p $HOME/miniforge3
source $HOME/miniforge3/etc/profile.d/conda.sh
```

**pure pip (Linux with sudo):**

```bash
sudo apt-get install -y build-essential libgsl-dev libfftw3-dev
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name ksz21 --display-name "Python (ksz21)"
```

**pure pip (macOS, Homebrew):**

```bash
brew install gsl fftw
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name ksz21 --display-name "Python (ksz21)"
```


