# XPCS_FitLab

XPCS_FitLab is a Python tool for analyzing X-ray Photon Correlation
Spectroscopy (XPCS) data stored in HDF5 files. It reads previously calculated
intensity autocorrelation functions, fits relaxation models and exports the
results.

**Version:** 7.0

## Features

- Read supported Sirius, APS and ESRF HDF5 formats.
- Select files, the fitting time interval and, for a single file, q curves.
- Fit single exponential, generalized exponential (stretched or compressed)
  and cumulants models.
- Calculate inverse rates, rate/q-squared values and relaxation-rate fits
  as a function of q.
- Save figures, fitted curves, parameter tables and an analysis record.

## Installation

Install the required packages in the Python environment used by Jupyter:

```bash
python3 -m pip install h5py numpy scipy scikit-learn "matplotlib>=3.7" pandas
python3 -m pip install notebook
```

The selection windows also require Tkinter and a local graphical session.
On Ubuntu, install Tkinter if needed:

```bash
sudo apt install python3-tk
```

To check Tkinter, run `python3 -m tkinter`; a small test window should appear.

## Use

1. Keep `XPCS_functions.py` and `hdf5_xpcs_fitlab.ipynb` in the same directory.
2. Open the notebook:

   ```bash
   jupyter notebook hdf5_xpcs_fitlab.ipynb
   ```

3. Review the `USER SETTINGS` block at the beginning of the code cell.
4. Run the cell and follow the format, file and data-selection windows.
5. Review the summary and exported results.

Restart the Jupyter kernel after replacing the functions module.

`hdf5_xpcs_fitlab.py` contains the same code as the notebook, exported with
`jupyter nbconvert --to script`. The notebook is the documented entry point.

Both time selectors support mouse dragging. Left/Right move the lower limit
and Down/Up move the upper limit, one position at a time.

The default R-squared threshold is 0.9. Fits with scores strictly above this
threshold and finite derived values are included in parameter averages and
global rate fits. Individual results are also saved for review.
Two-time matrix export and comparison with stored Sirius 3 fits are optional
and disabled by default.

## Outputs

For each input file, the program saves a two-page PDF and a results directory
with correlation data, fitted curves, parameters and an analysis record.
Combined parameter tables and the low-R-squared report are saved in a new
`_summary/run_NNN/` directory for each execution.

Results are saved in the input directory unless `out_dir` is specified.
PDFs and other fixed-name exports are overwritten when the same input is
analyzed again in the same output directory. Use a different `out_dir` to
keep the results of separate runs.

Input times are assumed to be in seconds and q in reciprocal angstroms.
Interpreting the derived coefficients as diffusion coefficients requires
the appropriate physical model and input units.

## Citation

Citation metadata are provided in `CITATION.cff`; GitHub shows them under
"Cite this repository". Version 6.7 is archived on Zenodo:
https://doi.org/10.5281/zenodo.13835703

## License

MIT. See `LICENSE`.
