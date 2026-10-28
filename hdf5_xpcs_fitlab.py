#!/usr/bin/env python
# coding: utf-8

# # HDF5 XPCS FitLab
# 
# **Tool for analyzing X-ray Photon Correlation Spectroscopy (XPCS) data stored in HDF5 files.**
# 
# **Author:** Marilina Cathcarth [[mcathcarth@gmail.com](mailto:mcathcarth@gmail.com)]
# 
# **Version:** 7.0
# 
# **Date:** October 2026
# 
# The program reads previously calculated correlation functions, fits relaxation models, analyzes their parameters and exports the results.
# 
# ## Requirements
# 
# - Python. Current synthetic-data checks were performed with Python 3.12.14.
# - h5py, NumPy, SciPy, scikit-learn, Matplotlib and pandas.
# - Tkinter for the selection windows.
# - Jupyter Notebook, JupyterLab or another compatible notebook interface.
# - A local graphical desktop session for the Tkinter windows.
# 
# Place `XPCS_functions.py` in the same directory as this notebook.
# Use matching versions of the notebook and the functions module.
# 
# The language-version metadata records the validation environment, not a minimum Python version.
# Record the actual Python and dependency versions used for the experimental example.
# 
# ## Installation
# 
# Install the scientific dependencies in the Python environment used by the notebook:
# 
# ```bash
# python -m pip install h5py numpy scipy scikit-learn matplotlib pandas
# ```
# 
# If you need Jupyter Notebook, install it with:
# 
# ```bash
# python -m pip install notebook
# ```
# 
# Then launch it with:
# 
# ```bash
# jupyter notebook
# ```
# 
# On Ubuntu, install Tkinter if it is missing:
# 
# ```bash
# sudo apt install python3-tk
# ```
# 
# To check Tkinter, run:
# 
# ```bash
# python -m tkinter
# ```
# 
# A small test window should appear.
# 
# ## User settings
# 
# The `USER SETTINGS` block at the beginning of the code cell contains the available configuration options.
# 
# The default settings use selection windows.
# You can also edit the settings before running the analysis:
# 
# | Setting | Purpose |
# |---|---|
# | `R2_threshold` | Select fits for parameter averages and relaxation-rate analyses. Accepted fits require R^2 strictly above this threshold and valid derived parameters. |
# | `directory` | Input directory. Leave `''` to select a directory or individual HDF5 files through a window. |
# | `out_dir` | Output directory. Leave `''` to save results in the input directory. |
# | `export_t1t2` | Export two-time matrices when available. Default: `False`. |
# | `compare_with_beamline` | Enable the optional Sirius 3 reference-fit diagnostic. Default: `False`. |
# | `selected_synchrotron` | Input format. Leave `None` to use the selection windows, or uncomment one manual format option. |
# | `data_type` | Data representation for `Sirius 3` and `Sirius 3 average`: `'Originals'` or `'Rebinneds'`. Other formats use `None`. |
# 
# Within `USER SETTINGS`:
# 
# - Instructions start with `##`.
# - Optional code lines start with a single `#`.
# - Keep the default assignments active.
# - For manual format selection, uncomment one format option.
# - For either Sirius 3 format, also uncomment one data representation.
# 
# ## Running the analysis
# 
# 1. Review the `USER SETTINGS` block.
# 2. Run the code cell.
# 3. Complete the selection dialogs that appear.
# 4. Review the processing summary and exported results.
# 
# During execution, you can choose whether to restrict the delay-time interval.
# When a single input file is selected, you can also select the q curves to analyze.
# 
# Time-range controls work in both the basic selector and the live preview:
# 
# - Left/Right move the lower limit by one stored time position.
# - Down/Up move the upper limit by one stored time position.
# - Mouse dragging remains available.
# - The live preview updates its labels and selected region after each move.
# - Modified widget shortcuts remain available; Alt+Down opens the q preview list.
# - While that list is open, its arrow keys navigate q values.
# 
# Apply Range confirms the selected positions. Closing the window, or choosing
# Use Full Range in the preview, retains the full input interval.
# 
# Beamline comparison is disabled by default. Set `compare_with_beamline = True`
# only to compare the program KWW parameters with stored Sirius 3 reference fits.
# With `False`, reference fit parameters are not read, differences are not calculated
# and comparison files are not written or modified. The program fits are unchanged.
# 
# 
# ## Parameter interpretation and units
# 
# Input delay times are assumed to be in seconds and q values in angstrom^-1.
# The readers retain the stored values; input units are not automatically validated or converted.
# 
# Per-curve rate/q^2 values are exported in um^2/s using a factor of 1e-8.
# They require an appropriate physical model to be interpreted as diffusion coefficients.
# 
# The fixed-exponent fit is C = D_app q^2.
# The free-exponent fit is C = K q^n, where K has units of angstrom^n/s.
# 
# Exported inverse rates are 1/C for the single and generalized exponential models and 1/C1 for cumulants.
# The cumulants inverse rate is not generally the time at which the relaxation function reaches 1/e.
# 
# ## Outputs
# 
# The output directory is defined by `out_dir`.
# When `out_dir` is empty, results are saved in the input directory.
# 
# | Location | Contents |
# |---|---|
# | Output directory | One two-page `<base>.pdf` report per processed input file. |
# | `<base>/` | Individual fit files, correlation DataFrame exports, analysis records and optional beamline comparisons when enabled. |
# | `<base>/t1-vs-t2/` | Two-time correlation matrices, when available. |
# | `_summary/run_001/`, `run_002/`, ... | Combined fit-result tables and the low-R^2 report for each execution. |
# 
# Each `<base>_analysis_<run_id>.dat` record contains the original input filename, software version, input format, data representation, R^2 threshold, beamline-comparison setting, requested time interval and q selection.
# Its per-q table records the number of points supplied to the models, actual time bounds, numerical status and the three fit scores.
# The same run identifier appears on both PDF pages.
# 
# The correlation DataFrame export retains the available delay-time rows.
# Individual fit files contain the selected and cleaned time-correlation pairs supplied to the models.
# 
# Analysis records are preserved separately for each execution.
# PDFs and other exports retain fixed filenames and are overwritten when the same input is analyzed again in the same output directory.

# In[ ]:


#=======================================================================#
#                                IMPORTS                                #
#=======================================================================#

# Standard library
import os

# Scientific libraries
import numpy as np

# XPCS analysis functions
from XPCS_functions import (
    # Directory and file handling
    AnalysisStopped, select_synchrotron, list_hdf5_files, select_directory_or_files,
    prepare_output_names, prepare_output_directory, prepare_summary_directory,
    ask_user_for_t_range, generate_alphabetic_suffix,
    # HDF5 readers
    read_hdf5_file,
    # Data cleaning and selection
    extract_q_values, clean_series, get_t_range_limits, get_time_bounds,
    ask_user_for_select_q, select_q_columns,
    # Data export
    save_dataframe_to_tsv, save_t1t2_data,
    # Initializations
    initialize_error_and_success_counters, initialize_data_for_parameter_averages,
    initialize_plot, initialize_data_for_derived_params,
    # Fit models and comparison
    fit_single_exponential, fit_stretched_exponential, fit_cumulants,
    compare_fit_parameters,
    # Calculations and data processing
    calculate_relaxation_and_diffusion, calculate_pdi, update_parameter_tracking,
    calculate_average_parameter_values,
    # Result saving and reporting
    initialize_analysis_record, record_analysis_curve, save_analysis_record,
    record_fit_status, add_data_to_table, write_dat_file, save_figures_to_pdf,
    save_fit_variations, print_summary, generate_fit_results_tables,
    # Plotting and visualization
    add_analysis_details, plot_data_and_curves, fit_and_plot_model, annotate_rate_fit,
    plot_params, configure_subplot, add_text_best_position,
)

## =================================================================== ##
##                           USER SETTINGS                             ##
## =================================================================== ##
## Edit the values below or uncomment the available options.
## Instructions start with ##.
## Optional code lines start with a single #.
## =================================================================== ##

## -------------------------- FIT SELECTION -------------------------- ##

## R^2 threshold for selecting fits for parameter averages.
## Include only fits with R^2 strictly greater than this threshold.
## Fits at or below the threshold are excluded from the averages.
## Parameter tracking also requires finite derived values.
R2_threshold = 0.9

## ---------------------- INPUT AND OUTPUT PATHS --------------------- ##

## Input directory: leave empty to open the selection window.
directory = ''

## Output directory: leave empty to save in the input directory.
out_dir = ''

## ------------------------- OPTIONAL EXPORTS ------------------------ ##

## Export two-time correlation matrices as text files.
## Set False to skip this export; g2 fitting is unchanged.
export_t1t2 = False

## ---------------------- OPTIONAL COMPARISON ----------------------- ##

## Compare with stored Sirius 3 reference fits for diagnostic checks.
## Set True only when this comparison is explicitly needed.
## False skips reference-fit reading, differences and comparison exports.
compare_with_beamline = False

## -------------------------- INPUT FORMAT --------------------------- ##

## Keep these defaults active.
## A manual selection below overrides these values.
selected_synchrotron = None
data_type = None

## Uncomment ONE format option below.
## Leave all options commented to use the selection windows.

## SIRIUS
## Version 1 (May, 2023)
#selected_synchrotron, data_type = 'Sirius 1', None

## Version 2 (October, 2023)
#selected_synchrotron, data_type = 'Sirius 2', None

## Version 3 (January, 2024)
#selected_synchrotron = 'Sirius 3'

## Version 3 (Average)
#selected_synchrotron = 'Sirius 3 average'

## For either Sirius 3 format, also uncomment ONE data representation.
## Original data:
#data_type = 'Originals'
## Multi-tau data:
#data_type = 'Rebinneds'

## APS
#selected_synchrotron, data_type = 'APS', None

## ESRF
#selected_synchrotron, data_type = 'ESRF', None

## =================================================================== ##
##                        END OF USER SETTINGS                         ##
## =================================================================== ##

#=======================================================================#
#                 INPUT SELECTION AND OUTPUT PREPARATION                #
#=======================================================================#

# Get the selected synchrotron and data representation when not set manually.
if selected_synchrotron is None:
    selected_synchrotron, data_type = select_synchrotron()

# Initialize the file list before directory selection.
hdf5_files = None

# Get HDF5 filenames from the manually defined directory.
if directory:
    hdf5_files = list_hdf5_files(directory)

# Select a directory or files when no directory was provided
# or the manually defined directory could not be read.
if hdf5_files is None:
    directory, hdf5_files = select_directory_or_files()

# Stop if no HDF5 files were selected or found.
if not hdf5_files:
    raise AnalysisStopped("No HDF5 files selected or found.")
    
# Prepare distinct output names before processing any files.
output_names_by_file = prepare_output_names(selected_synchrotron, hdf5_files)

# Create and check the output directory.
# Use the input directory when out_dir is empty.
output_directory = prepare_output_directory(directory, out_dir)

# Create a separate summary directory for this execution.
summary_directory = prepare_summary_directory(output_directory)

#=======================================================================#
#                          BATCH INITIALIZATION                         #
#=======================================================================#

# Initialize the dictionary of error and success counters.
counters = initialize_error_and_success_counters()

# Initialize empty lists for table data.
table_data = {
    "single": [],     # Empty list for single table data
    "stretched": [],  # Empty list for stretched table data
    "cumulants": []   # Empty list for cumulants table data
}

# Select a delay-time interval using the first dataset with a usable interval.
# Reuse the selected time bounds for subsequent files.
first_iteration = True

# Ask once whether to select a delay-time interval for fitting.
define_t_range = ask_user_for_t_range()

#=======================================================================#
#                          PROCESS INPUT FILES                          #
#=======================================================================#

## Loop over each .hdf5 file ##

for hdf5_file in hdf5_files:
    file_path = os.path.join(directory, hdf5_file)

    counters['hdf5_files'] += 1    # Increment the hdf5_file counter
    
    #-------------------------------------------------------------------#
    #                       READ CORRELATION DATA                       #
    #-------------------------------------------------------------------#

    # Obtain g2 vs. t data for each q value and create a DataFrame #
    dataset_df, t1t2_dfs, fit_parameters_dict = read_hdf5_file(
        file_path, selected_synchrotron, data_type,
        compare_with_beamline=compare_with_beamline)

    # Record and skip files whose correlation curves could not be read.
    if dataset_df is None:
        counters['invalid_data'] += 1
        counters['invalid_files'].append(f"{hdf5_file}: correlation read failed")
        continue

    # Retrieve the output base name prepared for this input file.
    base_name = output_names_by_file[hdf5_file]

    # Initialize the comparison rows for the current file.
    fit_variations = []
    
    #-------------------------------------------------------------------#
    #                         PREPARE q COLUMNS                         #
    #-------------------------------------------------------------------#
    
    # Extract q values and sort the correlation columns numerically.
    q_values_by_column = extract_q_values(dataset_df)
    q_columns_df = sorted(
        q_values_by_column, key=q_values_by_column.get)
    
    # Record and skip files without recognizable correlation columns.
    if not q_columns_df:
        counters['invalid_data'] += 1
        counters['invalid_files'].append(f"{hdf5_file}: no recognizable g2 columns")

        print(f"Warning: No valid g2 columns were found in '{hdf5_file}'. Skipping.")
        continue

    # Reorder the DataFrame columns, keeping '# t' at the beginning.
    dataset_df = dataset_df[['# t'] + q_columns_df]

    # +++ Increment the q_values counter +++
    q_count = len(q_columns_df)  # Correlation curves before q selection.
    counters['q_values'] += q_count
    
    #-------------------------------------------------------------------#
    #                   SELECT THE DELAY-TIME INTERVAL                  #
    #-------------------------------------------------------------------#

    # Define the range once, using the first valid dataset.
    # Single input file: graphical preview.
    # Multiple input files: simple slider only once.
    if first_iteration:
        show_range_preview = len(hdf5_files) == 1

        lower_limit, upper_limit = get_t_range_limits(
            dataset_df, define_t_range, show_preview=show_range_preview)

        # Keep time bounds for all files, rather than shared row positions.
        time_range = None
        if define_t_range:
            # Add 1 to include the upper limit.
            selected_times = dataset_df['# t'].iloc[lower_limit:upper_limit + 1]
            time_range = get_time_bounds(selected_times)
            
            # Record this file and its curves when the interval has no finite times.
            # Try range selection again with the next file.
            if time_range is None:
                counters['invalid_data'] += 1
                counters['invalid_files'].append(
                    f"{hdf5_file}: no finite times in the selected interval")
                counters['skipped_q'] += q_count
                continue

        first_iteration = False  # Set the flag to False.

    #-------------------------------------------------------------------#
    #               PREPARE EXPORT PATHS AND TWO-TIME DATA              #
    #-------------------------------------------------------------------#
    
    # Create the data directory for the current input file.
    file_output_directory = prepare_output_directory(
        os.path.join(output_directory, base_name))

    # Define the final correlation-DataFrame output path.
    output_dataframe = os.path.join(
        file_output_directory, f"{base_name}_export_DataFrame.dat")
    
    # Export available two-time matrices when requested.
    if export_t1t2 and t1t2_dfs:
        # Save available t1-t2 matrices inside the current file's data directory.
        save_t1t2_data(t1t2_dfs, file_output_directory, base_name)

    #-------------------------------------------------------------------#
    #              INITIALIZE PER-FILE RESULTS AND FIGURES              #
    #-------------------------------------------------------------------#
    
    # Initialize data structures for parameter averages
    parameter_averages_data = initialize_data_for_parameter_averages()
    
    # Call the function to initialize the plot
    (fig1, (ax_single, ax_stretched, ax_cumulants, ax_Cq2_si, ax_Cq_si), 
    fig2, (ax_Cq2_st, ax_Cq_st, ax_gammaq, ax_Cq2_cu, ax_Cq_cu, ax_PDIq), 
    cmap, lines_labels_dict) = initialize_plot(q_count)

    # Initialize empty lists for derived and qualified parameters
    derived_params, qualified_params = initialize_data_for_derived_params()
    
    # Keep each column's color before optional q selection.
    q_colors_by_column = {column: cmap(index)
                          for index, column in enumerate(dataset_df.columns[1:])}
    
    #-------------------------------------------------------------------#
    #                 SELECT q COLUMNS FOR A SINGLE FILE                #
    #-------------------------------------------------------------------#

    # If single file:
    if len(hdf5_files) == 1:

        # Ask the user if they want to select q values
        select_q_values = ask_user_for_select_q()

        if select_q_values:
            # Create a copy of the dataset
            original_dataset_df = dataset_df.copy()

            # Open the selection window and get the resulting dataset
            dataset_df = select_q_columns(
                dataset_df, lower_limit, upper_limit, cmap)

            # Save the original DataFrame as tab-separated text.
            # Use the current file's data directory.
            output_df_original = os.path.join(
                file_output_directory, f"{base_name}_export_DForiginal.dat")
            # Save DataFrame to a TSV file
            save_dataframe_to_tsv(original_dataset_df, output_df_original)
    
    #-------------------------------------------------------------------#
    
    # Count curves removed by the optional q selection
    counters['excluded_q'] += q_count - (len(dataset_df.columns) - 1)
    
    # Record settings and q selection for the current input file.
    analysis_record = initialize_analysis_record(
        file_path, base_name, selected_synchrotron, data_type,
        R2_threshold, time_range, q_values_by_column, dataset_df.columns[1:],
        compare_with_beamline=compare_with_beamline)
    
    #-------------------------------------------------------------------#
    #                       PROCESS EACH q SERIES                       #
    #-------------------------------------------------------------------#
    
    ## Loop over each q value ##
    
    for i, q_column in enumerate(dataset_df.columns[1:], 1):
        
        #---------------- PREPARE THE CURRENT q SERIES -----------------#

        # Retrieve the q value already extracted from the column name
        q_value = q_values_by_column[q_column]

        # Get the 't' and 'g2' data for the current q value
        t = dataset_df['# t']
        g2 = dataset_df[q_column]

        # Apply the selected time interval once, before cleaning
        if time_range is not None:
            t_min, t_max = time_range

            # Include both limits and keep the time-correlation pairs aligned
            selected_rows = (t >= t_min) & (t <= t_max)
            t = t.loc[selected_rows]
            g2 = g2.loc[selected_rows]

        # Remove nonfinite pairs and a leading zero-delay sample
        # Require at least eight remaining points
        t, g2 = clean_series(t, g2, min_points=8, return_pandas=True)
        
        # Count and skip curves with too few points after cleaning
        if t is None:
            record_analysis_curve(analysis_record, q_value)
            counters['skipped_q'] += 1
            print(f"Warning: Skipping '{hdf5_file}', q = {q_value}: "
                  "fewer than 8 valid points in the analysis interval.")
            continue

        # Create the output filename, continuing with aa, ab, ... after z
        suffix = generate_alphabetic_suffix(i)
        output_name = f"{base_name}_export_{suffix}.dat"
        
        # Save individual q results in the current file's data directory.
        output_path = os.path.join(file_output_directory, output_name)

        
        #---------------------------------------------------------------#
        #-------------------- FIT THE THREE MODELS ---------------------#

        #----- Fit the Single Exponential Model -----#

        # Fit A, B and C in the Single Exponential Model
        fit_params_single, fitted_curve_single, r2_single = (
            fit_single_exponential(t, g2))

        # Calculate relaxation time and C/q^2 for Single Exponential
        relax_time_single, diffusion_coef_single = (
            calculate_relaxation_and_diffusion(fit_params_single, q_value))

        #----- Fit the Stretched Exponential model -----#

        # Fit C and gamma with A and B fixed from the Single Exponential fit
        fit_params_stretched, fitted_curve_stretched, r2_stretched = (
            fit_stretched_exponential(t, g2, fit_params_single))

        # Calculate relaxation time and C/q^2 for Stretched Exponential
        relax_time_stretched, diffusion_coef_stretched = (
            calculate_relaxation_and_diffusion(fit_params_stretched, q_value))

        #----- Analysis by the method of Cumulants -----#

        # Fit C1 and C2 with A and B fixed from the Single Exponential fit
        fit_params_cumulants, fitted_curve_cumulants, r2_cumulants = (
            fit_cumulants(t, g2, fit_params_single))

        # Calculate relaxation time and C1/q^2 for Cumulants
        relax_time_cumulants, diffusion_coef_cumulants = (
            calculate_relaxation_and_diffusion(fit_params_cumulants, q_value))

        #-----     -----     -----     -----     -----     -----#
        # Record the numerical status of this q series once
        record_fit_status(
            counters, hdf5_file, q_value, (r2_single, r2_stretched, r2_cumulants))
        
        # Record the cleaned times actually supplied to the models.
        record_analysis_curve(
            analysis_record, q_value, t, (r2_single, r2_stretched, r2_cumulants))

        #---------------- STORE AND EXPORT FIT RESULTS -----------------#

        # Retrieve the Single Exponential parameters: A, B, C.
        A_single, B_single, C_single = fit_params_single

        # Append data to the table for Single Exponential.
        add_data_to_table(
            table_data["single"], base_name, q_value, fit_params_single,
            relax_time_single, diffusion_coef_single, r2_single)

        # Retrieve the Stretched Exponential parameters: A, B, C, gamma.
        A_stretched, B_stretched, C_stretched, gamma_stretched = fit_params_stretched

        # Append data to the table for Stretched Exponential.
        add_data_to_table(
            table_data["stretched"], base_name, q_value, fit_params_stretched,
            relax_time_stretched, diffusion_coef_stretched, r2_stretched)

        # Retrieve the Cumulants parameters: A, B, C1, C2.
        A_cumulants, B_cumulants, C1_cumulants, C2_cumulants = fit_params_cumulants

        # Calculate PDI from the cumulants, returning NaN when undefined.
        PDI_cumulants = calculate_pdi(C1_cumulants, C2_cumulants)

        # Append data to the table for Cumulants.
        # The summary table stores PDI in place of C2.
        add_data_to_table(
            table_data["cumulants"], base_name, q_value,
            [A_cumulants, B_cumulants, C1_cumulants, PDI_cumulants],
            relax_time_cumulants, diffusion_coef_cumulants, r2_cumulants)

        # Write data and results for all three models to the individual file.
        # Pass C2 to reconstruct the cumulants curve.
        write_dat_file(
            output_path=output_path, q_value=q_value, t=t, g2=g2,

            # Single Exponential results.
            A_single=A_single, B_single=B_single, C_single=C_single,
            r2_single=r2_single, relax_time_single=relax_time_single,
            diffusion_coef_single=diffusion_coef_single,

            # Stretched Exponential results.
            A_stretched=A_stretched, B_stretched=B_stretched,
            C_stretched=C_stretched, gamma=gamma_stretched,
            r2_stretched=r2_stretched,
            relax_time_stretched=relax_time_stretched,
            diffusion_coef_stretched=diffusion_coef_stretched,

            # Cumulants results.
            A_cumulants=A_cumulants, B_cumulants=B_cumulants,
            C1_cumulants=C1_cumulants, C2_cumulants=C2_cumulants,
            r2_cumulants=r2_cumulants,
            relax_time_cumulants=relax_time_cumulants,
            diffusion_coef_cumulants=diffusion_coef_cumulants
        )

        #----------------- COLLECT DERIVED PARAMETERS ------------------#

        # Store q and q^2 for the relaxation-rate analyses.
        derived_params["q_value"].append(q_value)
        derived_params["q_square"].append(q_value**2)

        # Store relaxation rates and model-specific parameters.
        derived_params["relax_rate_single"].append(C_single)
        derived_params["relax_rate_stretched"].append(C_stretched)
        derived_params["gamma"].append(gamma_stretched)
        derived_params["relax_rate_cumulants"].append(C1_cumulants)
        derived_params["PDI"].append(PDI_cumulants)
        
        # Store the original color of this processed curve.
        derived_params["color"].append(q_colors_by_column[q_column])

        #------------------- PLOT CORRELATION CURVES -------------------#

        # Keep the color assigned before optional q selection.
        color = q_colors_by_column[q_column]
        #print(f"i: {i}, color: {color}")

        # Plot the Single Exponential results.
        plot_data_and_curves(
            ax_single, t, g2, A_single, B_single, fitted_curve_single,
            q_value, r2_single, lines_labels_dict, color,
            linestyle='-', model_type="Single")

        # Plot the Stretched Exponential results.
        plot_data_and_curves(
            ax_stretched, t, g2, A_stretched, B_stretched,
            fitted_curve_stretched, q_value, r2_stretched,
            lines_labels_dict, color,
            linestyle='--', model_type="Stretched")

        # Plot the Cumulants results.
        plot_data_and_curves(
            ax_cumulants, t, g2, A_cumulants, B_cumulants,
            fitted_curve_cumulants, q_value, r2_cumulants,
            lines_labels_dict, color,
            linestyle='dotted', model_type="Cumulants")
        

        #------------- SELECT FITS FOR PARAMETER AVERAGES --------------#

        #-- Collect parameters from fits with R^2 > R2_threshold.

        # Check that Single Exponential R^2 is finite and above the threshold.
        if np.isfinite(r2_single) and r2_single > R2_threshold:
            fit_included = update_parameter_tracking(
                parameter_averages_data, 'single', diffusion_coef_single)

            # Keep the selected lists aligned with the accepted averages.
            if fit_included:
                qualified_params["q_value_single"].append(q_value)
                qualified_params["relax_rate_single"].append(C_single)
                qualified_params["color_single"].append(color)

        # Record finite scores at or below the threshold.
        elif np.isfinite(r2_single):
            counters['below_threshold'] += 1
            counters['below_threshold_files'].append(
                f"{hdf5_file} - q = {q_value} A^-1 - "
                f"Single Exponential - R2 = {r2_single:.3f}")

        # Check that Stretched Exponential R^2 is finite and above the threshold.
        if np.isfinite(r2_stretched) and r2_stretched > R2_threshold:
            fit_included = update_parameter_tracking(
                parameter_averages_data, 'stretched',
                diffusion_coef_stretched, gamma_stretched)

            # Keep the selected lists aligned with the accepted averages.
            if fit_included:
                qualified_params["q_value_stretched"].append(q_value)
                qualified_params["relax_rate_stretched"].append(C_stretched)
                qualified_params["color_stretched"].append(color)
                qualified_params["gamma"].append(gamma_stretched)

        # Record finite scores at or below the threshold.
        elif np.isfinite(r2_stretched):
            counters['below_threshold'] += 1
            counters['below_threshold_files'].append(
                f"{hdf5_file} - q = {q_value} A^-1 - "
                f"Stretched Exponential - R2 = {r2_stretched:.3f}")

        # Check that Cumulants R^2 is finite and above the threshold.
        if np.isfinite(r2_cumulants) and r2_cumulants > R2_threshold:
            fit_included = update_parameter_tracking(
                parameter_averages_data, 'cumulants',
                diffusion_coef_cumulants, PDI_cumulants)

            # Keep the selected lists aligned with the accepted averages.
            if fit_included:
                qualified_params["q_value_cumulants"].append(q_value)
                qualified_params["relax_rate_cumulants"].append(C1_cumulants)
                qualified_params["color_cumulants"].append(color)
                qualified_params["PDI"].append(PDI_cumulants)

        # Record finite scores at or below the threshold.
        elif np.isfinite(r2_cumulants):
            counters['below_threshold'] += 1
            counters['below_threshold_files'].append(
                f"{hdf5_file} - q = {q_value} A^-1 - "
                f"Cumulants - R2 = {r2_cumulants:.3f}")

        #----------------- COMPARE WITH BEAMLINE FITS ------------------#

        # Compare only when explicitly enabled and references are available.
        if compare_with_beamline and fit_parameters_dict:
            # Retrieve the reference parameters for the current q value.
            beamline_fit_params = fit_parameters_dict.get(q_value)
            # Extract code-generated fit parameters.
            code_fit_params = {
                'Baseline': A_stretched,
                'Beta': B_stretched,
                'Relax_time': relax_time_stretched,
                'Gamma': gamma_stretched}

            # Perform the comparison.
            variation = compare_fit_parameters(
                q_value, beamline_fit_params, code_fit_params)

            # Append the variation to the list.
            fit_variations.append(variation)
            
        #---------------------------------------------------------------#
    
    #-------------------------------------------------------------------#
    #                   FIT RELAXATION RATES VERSUS q                   #
    #-------------------------------------------------------------------#

    #----- Linear fit: C = D_app * q^2, with zero intercept -----#

    # Fit and plot C versus q^2 for each model, with zero intercept.
    D_single_linear, n_single_linear, r_single_linear, std_err_single_linear = (
        fit_and_plot_model(qualified_params["q_value_single"],
                           qualified_params["relax_rate_single"],
                           ax_Cq2_si, qualified_params["color_single"],
                           "Single - Linear", model_type='linear'))

    D_stretched_linear, n_stretched_linear, r_stretched_linear, std_err_stretched_linear = (
        fit_and_plot_model(qualified_params["q_value_stretched"],
                           qualified_params["relax_rate_stretched"],
                           ax_Cq2_st, qualified_params["color_stretched"],
                           "Stretched - Linear", model_type='linear'))

    D_cumulants_linear, n_cumulants_linear, r_cumulants_linear, std_err_cumulants_linear = (
        fit_and_plot_model(qualified_params["q_value_cumulants"],
                           qualified_params["relax_rate_cumulants"],
                           ax_Cq2_cu, qualified_params["color_cumulants"],
                           "Cumulants - Linear", model_type='linear'))

    # Set labels and annotate the three linear fits.
    for ax, coefficient, exponent, r_metric, std_err in (
        (ax_Cq2_si, D_single_linear, n_single_linear,
         r_single_linear, std_err_single_linear),
        (ax_Cq2_st, D_stretched_linear, n_stretched_linear,
         r_stretched_linear, std_err_stretched_linear),
        (ax_Cq2_cu, D_cumulants_linear, n_cumulants_linear,
         r_cumulants_linear, std_err_cumulants_linear)
    ):
        annotate_rate_fit(
            ax, coefficient, exponent, r_metric, std_err, model_type='linear')

    #----- Power-law fit: C = K * q^n -----#

    # Fit and plot the power law for each model.
    # Keep the historical 'exponential' identifier expected by the function.
    D_single_exponential, n_single_exponential, r_single_exponential, std_err_single_exponential = (
        fit_and_plot_model(qualified_params["q_value_single"],
                           qualified_params["relax_rate_single"],
                           ax_Cq_si, qualified_params["color_single"],
                           "Single - Power law", model_type='exponential'))

    D_stretched_exponential, n_stretched_exponential, r_stretched_exponential, std_err_stretched_exponential = (
        fit_and_plot_model(qualified_params["q_value_stretched"],
                           qualified_params["relax_rate_stretched"],
                           ax_Cq_st, qualified_params["color_stretched"],
                           "Stretched - Power law", model_type='exponential'))

    D_cumulants_exponential, n_cumulants_exponential, r_cumulants_exponential, std_err_cumulants_exponential = (
        fit_and_plot_model(qualified_params["q_value_cumulants"],
                           qualified_params["relax_rate_cumulants"],
                           ax_Cq_cu, qualified_params["color_cumulants"],
                           "Cumulants - Power law", model_type='exponential'))

    # Set labels and annotate the three power-law fits.
    for ax, coefficient, exponent, r_metric, std_err in (
        (ax_Cq_si, D_single_exponential, n_single_exponential,
         r_single_exponential, std_err_single_exponential),
        (ax_Cq_st, D_stretched_exponential, n_stretched_exponential,
         r_stretched_exponential, std_err_stretched_exponential),
        (ax_Cq_cu, D_cumulants_exponential, n_cumulants_exponential,
         r_cumulants_exponential, std_err_cumulants_exponential)
    ):
        annotate_rate_fit(
            ax, coefficient, exponent, r_metric, std_err, model_type='exponential')
    
    #-------------------------------------------------------------------#
    #                        PLOT GAMMA AND PDI                         #
    #-------------------------------------------------------------------#

    #----- Gamma vs q -----#

    # Plot all valid gamma values using the stored curve colors.
    plot_params(derived_params["q_value"], derived_params["gamma"],
                ax_gammaq, cmap, "gamma", colors=derived_params["color"])

    # Set subplot labels.
    ax_gammaq.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)", fontsize=14)
    ax_gammaq.set_ylabel(r"$\gamma$", fontsize=14)
    ax_gammaq.set_title(r"$\gamma$ vs $q$", fontsize=16)

    # Use the same selected records and statistics as the parameter averages.
    average_stretched_parameters = calculate_average_parameter_values(
        parameter_averages_data, 'stretched')
    average_gamma = average_stretched_parameters['Gamma av']
    stdev_gamma = average_stretched_parameters['Gamma std']

    # Add the mean and population standard deviation when records are available.
    if np.isfinite(average_gamma):
        add_text_best_position(ax_gammaq,
                               f"Mean gamma = {average_gamma:.3g}\n"
                               f"SD = {stdev_gamma:.3g}",
                               fontsize=12)

    #----- Plot PDI vs q -----#

    # Plot all valid PDI values using the stored curve colors.
    plot_params(derived_params["q_value"], derived_params["PDI"],
                ax_PDIq, cmap, "PDI", colors=derived_params["color"])

    # Set subplot labels.
    ax_PDIq.set_xlabel(r"$q$ ($\mathrm{\AA}^{-1}$)", fontsize=14)
    ax_PDIq.set_ylabel("PDI", fontsize=14)
    ax_PDIq.set_title(r"PDI vs $q$", fontsize=16)

    # Use the same selected records and statistics as the parameter averages.
    average_cumulants_parameters = calculate_average_parameter_values(
        parameter_averages_data, 'cumulants')
    average_PDI = average_cumulants_parameters['PDI av']
    stdev_PDI = average_cumulants_parameters['PDI std']

    # Add the mean and population standard deviation when records are available.
    if np.isfinite(average_PDI):
        add_text_best_position(ax_PDIq,
                               f"Mean PDI = {average_PDI:.3g}\n"
                               f"SD = {stdev_PDI:.3g}",
                               fontsize=12)
    
    #-------------------------------------------------------------------#
    #                  CONFIGURE CORRELATION SUBPLOTS                   #
    #-------------------------------------------------------------------#

    # Configure subplot for Single Exponential.
    configure_subplot(ax_single, "Single Exponential", "Delay time (s)",
                      r"$(g_2-\mathrm{baseline})/\beta$",
                      lines_labels_dict["exp_lines_single"],
                      lines_labels_dict["exp_labels_single"],
                      lines_labels_dict["fit_lines_single"],
                      lines_labels_dict["fit_labels_single"])

    # Calculate parameter averages for the Single Exponential model.
    average_single_parameters = calculate_average_parameter_values(
        parameter_averages_data, 'single')

    # Add the mean and standard deviation if at least one fit was included.
    if parameter_averages_data['single']['fit_count'] > 0:
        add_text_best_position(ax_single,
            r"Mean $C/q^2$ ($\mu\mathrm{m}^2/\mathrm{s}$):" "\n"
            f"{average_single_parameters['Diff_coef av']:.4g} "
            f"({average_single_parameters['Diff_coef std']:.4g})",
            fontsize=12)

    # Configure subplot for Stretched Exponential.
    configure_subplot(ax_stretched, "Stretched Exponential", "Delay time (s)",
                      r"$(g_2-\mathrm{baseline})/\beta$",
                      lines_labels_dict["exp_lines_stretched"],
                      lines_labels_dict["exp_labels_stretched"],
                      lines_labels_dict["fit_lines_stretched"],
                      lines_labels_dict["fit_labels_stretched"])

    # Reuse the averages calculated in the gamma-versus-q block.
    # Add the mean and standard deviation if at least one fit was included.
    if parameter_averages_data['stretched']['fit_count'] > 0:
        add_text_best_position(ax_stretched,
            r"Mean $C/q^2$ ($\mu\mathrm{m}^2/\mathrm{s}$): "
            f"{average_stretched_parameters['Diff_coef av']:.4g} "
            f"({average_stretched_parameters['Diff_coef std']:.4g})\n"
            f"Av. Gamma: {average_stretched_parameters['Gamma av']:.4g} "
            f"({average_stretched_parameters['Gamma std']:.4g})",
            fontsize=12)

    # Configure subplot for Cumulants.
    configure_subplot(ax_cumulants, "Cumulants", "Delay time (s)",
                      r"$(g_2-\mathrm{baseline})/\beta$",
                      lines_labels_dict["exp_lines_cumulants"],
                      lines_labels_dict["exp_labels_cumulants"],
                      lines_labels_dict["fit_lines_cumulants"],
                      lines_labels_dict["fit_labels_cumulants"])

    # Reuse the averages calculated in the PDI-versus-q block.
    # Add the mean and standard deviation if at least one fit was included.
    if parameter_averages_data['cumulants']['fit_count'] > 0:
        add_text_best_position(ax_cumulants,
            r"Mean $C_1/q^2$ ($\mu\mathrm{m}^2/\mathrm{s}$): "
            f"{average_cumulants_parameters['Diff_coef av']:.4g} "
            f"({average_cumulants_parameters['Diff_coef std']:.4g})\n"
            f"Av. PDI: {average_cumulants_parameters['PDI av']:.4g} "
            f"({average_cumulants_parameters['PDI std']:.4g})",
            fontsize=12)
    
    #-------------------------------------------------------------------#
    #                       SAVE PER-FILE OUTPUTS                       #
    #-------------------------------------------------------------------#
    
    # Save a separate analysis record for this input and execution.
    analysis_record_path = save_analysis_record(analysis_record, file_output_directory)

    # Identify the same analysis on both PDF pages.
    add_analysis_details([fig1, fig2], analysis_record)
    
    # Save the PDF in the main output directory.
    # Save the two figures as separate PDF pages and close them.
    pdf_name = f"{base_name}.pdf"
    output_pdf_path = os.path.join(output_directory, pdf_name)
    save_figures_to_pdf([fig1, fig2], output_pdf_path)

    # Save the final DataFrame, including any q-column selection.
    # The columns retain the numerical q order established earlier.
    save_dataframe_to_tsv(dataset_df, output_dataframe)

    # Write diagnostic comparisons only when enabled for Sirius 3.
    # Disabled comparison leaves any previous comparison file unchanged.
    if compare_with_beamline and selected_synchrotron == 'Sirius 3':
        variations_output_path = os.path.join(
            file_output_directory, f"{base_name}_fit_variations.dat")
        save_fit_variations(fit_variations, variations_output_path)

#=======================================================================#
#                             FINAL REPORTS                             #
#=======================================================================#

# Print the processing summary and save the low-R^2 report.
print_summary(counters, directory=summary_directory, r2_threshold=R2_threshold)

# Generate the combined fit-result tables for all processed files.
generate_fit_results_tables(summary_directory,
                            table_data["single"],
                            table_data["stretched"],
                            table_data["cumulants"])

