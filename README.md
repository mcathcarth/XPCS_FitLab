# XPCS_FitLab

**XPCS_FitLab** is a toolset designed for analyzing X-ray Photon Correlation Spectroscopy (XPCS) data stored in HDF5 files. The tool includes both a Jupyter notebook (`hdf5_xpcs_fitlab.ipynb`) and a Python script (`hdf5_xpcs_fitlab.py`), both of which provide the same functionality. They offer a graphical interface (GUI) for managing input files and analyzing multiple HDF5 datasets efficiently.

The tool **relies on functions defined in the `XPCS_functions.py` file**, which must be located in the same directory as the main script or notebook for proper execution.

## Features

- **HDF5 Data Handling**: Reads XPCS data from HDF5 files using the `h5py` library.
- **Autocorrelation Function Fitting**: Fits precomputed $g_2(t)$ functions from the HDF5 files to three different models:
  - Single Exponential
  - Stretched Exponential
  - Cumulant Method
- **Diffusion Coefficient Calculation**: Calculates the diffusion coefficient ($D_0$) using two methods:
  - Direct calculation from the relaxation rate ($C$).
  - Linear fit of $C$ vs $q²$.
- **Graphical Interface**: Provides a GUI for selecting input files, setting parameters, and visualizing results.

## Installation

Before running **XPCS_FitLab**, ensure you have Python 3.6 or later installed. You can download it from [Python Downloads](https://www.python.org/downloads/).

To install the required Python packages, run the following command in your terminal or command prompt:

```bash
pip install h5py numpy scipy scikit-learn matplotlib qtpy pandas PyQt5 tk
```

## How to Use

### Running with Jupyter Notebook

1. Open `hdf5_xpcs_fitlab.ipynb` in Jupyter Notebook or JupyterLab.
2. Follow the instructions within the notebook to configure and run your analysis.

### Running with Python Script

1. Ensure that `hdf5_xpcs_fitlab.py` and `XPCS_functions.py` are located in the same directory.
2. Run `hdf5_xpcs_fitlab.py` from the terminal:

    ```bash
    python hdf5_xpcs_fitlab.py
    ```

    The GUI will open, allowing you to select files and perform the analysis.

## Important Notes

- The `XPCS_functions.py` file contains all the necessary functions used by both the notebook and script. Make sure it is located in the same directory as `hdf5_xpcs_fitlab.py` for the script to work correctly.
- You can define the $R²$ threshold for selecting fitted curves to include in the diffusion coefficient calculations. Both the direct calculation and linear fit methods will use this threshold.

## Dependencies

- Python (>= 3.6)
- h5py
- numpy
- scipy
- scikit-learn
- matplotlib
- qtpy
- pandas
- PyQt5
- tk

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

