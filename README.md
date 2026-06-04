# XPCS_FitLab

**XPCS_FitLab** is a toolset designed for analyzing X-ray Photon Correlation Spectroscopy (XPCS) data stored in HDF5 files.

The tool includes both a Jupyter notebook (`hdf5_xpcs_fitlab.ipynb`) and a Python script (`hdf5_xpcs_fitlab.py`). Both rely on the functions defined in `XPCS_functions.py`, which must be located in the same directory as the notebook or script.

## Current Version

**Version:** 6.8
**Release date:** June 2026

## Features

* **HDF5 Data Handling**: Reads XPCS data from HDF5 files using the `h5py` library.
* **Autocorrelation Function Fitting**: Fits precomputed (g_2(t)) functions from HDF5 files using three models:

  * Single Exponential
  * Stretched Exponential
  * Cumulants Method
* **Diffusion Coefficient Calculation**: Calculates diffusion coefficients using:

  * Direct calculation from the relaxation rate (C).
  * Linear fit of (C) vs (q^2).
  * Power-law fit of (C) vs (q).
* **Graphical Interface**:

  * Select individual HDF5 files or complete directories.
  * Define the fitting time range.
  * Preview the selected time range when processing a single file.
  * Select or exclude specific (q) values interactively.
* **Output Files**:

  * PDF reports with fitted curves and derived-parameter plots.
  * `.dat` files with fitted curves and fit parameters.
  * Summary tables for each fitting model.
  * A report listing fits below the selected (R^2) threshold.

## Installation

Before running **XPCS_FitLab**, make sure Python 3.8 or later is installed.

Install the required Python packages with:

```bash
pip install h5py numpy scipy scikit-learn matplotlib pandas
```

Tkinter is also required for the graphical interface. It is usually included with Python. On some Linux systems, it may need to be installed separately:

```bash
sudo apt install python3-tk
```

## Dependencies

* Python (>= 3.8)
* h5py
* numpy
* scipy
* scikit-learn
* matplotlib
* pandas
* tkinter

## How to Use

### Running with Jupyter Notebook

1. Make sure `hdf5_xpcs_fitlab.ipynb` and `XPCS_functions.py` are located in the same directory.
2. Open `hdf5_xpcs_fitlab.ipynb` in Jupyter Notebook or JupyterLab.
3. Run the notebook cells and follow the graphical interface prompts.

### Running with Python Script

1. Make sure `hdf5_xpcs_fitlab.py` and `XPCS_functions.py` are located in the same directory.
2. Run the script from the terminal:

```bash
python hdf5_xpcs_fitlab.py
```

The graphical interface will open, allowing you to select files or directories and perform the analysis.

## Important Notes

* The `XPCS_functions.py` file contains the functions used by both the notebook and the script.
* The `R2_threshold` parameter defines the minimum (R^2) value required for a fit to be included in averages and diffusion-related plots.
* Fits below the `R2_threshold` may still appear in the generated correlation-function plots and fit-results tables for diagnostic purposes.
* Fits that converge numerically but do not pass the (R^2) threshold are listed in `below_R2_threshold_fits.dat`.
* If there are not enough valid fits after applying the (R^2) threshold, diffusion-related plots are skipped instead of forcing invalid calculations.

## Version 6.8 Updates

* Added graphical time-range preview for single-file processing.
* Improved interactive (q)-value selection:

  * Curves start visible by default.
  * Unchecked (q) values are hidden from the preview plot.
  * Labels include colored markers matching the plotted curves.
* Improved Tkinter window handling.
* Replaced mixed PyQt/Tkinter file selection with a Tkinter-based selector.
* Improved automatic legend placement in PDF reports.
* Added automatic placement of average-parameter annotations to reduce overlap with curves and legends.
* Corrected the exported stretched-exponential fitted curve.
* Improved robustness when fits fail, return NaN values, or cannot be used for diffusion-related plots.
* Added a separate report for fits below the selected (R^2) threshold.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
