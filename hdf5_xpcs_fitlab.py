#!/usr/bin/env python
# coding: utf-8

# # HDF5 XPCS FitLab - Script for analyzing X-ray Photon Correlation Spectroscopy (XPCS) data from HDF5 files.
# 
# **Author:** Marilina Cathcarth [mcathcarth@gmail.com]
# 
# **Version:** 6.7
# 
# **Date:** September 3, 2024
# 
# **Important Note:**
# 
# This script relies on functions defined in the 'XPCS_functions.py' file. Please ensure that 'XPCS_functions.py' is located in the same directory as this script for proper execution.
# 
# **Dependencies:**
# 
# - Python (version >= 3.6)
# - h5py (to work with HDF5 files)
# - numpy (for numerical calculations)
# - scipy (for curve fitting)
# - scikit-learn (for calculating R2 score)
# - matplotlib (for data visualization)
# - qtpy (for GUI-based directory selection)
# - pandas (for data manipulation)
# - PyQt5 (for GUI)
# - tk (for GUI)
# - functools (for function manipulation)
#   
# **Installation:**
# 
# 1. Make sure you have Python 3.6 or later installed. If not, download and install Python from [Python Downloads](https://www.python.org/downloads/).
# 
# 2. Install the required packages using pip. Open a terminal or command prompt and run the following command:
# 
#     ```bash
#     pip install h5py numpy scipy scikit-learn matplotlib qtpy pandas PyQt5 tk
#     ```
#     .
# 4. Place the 'XPCS_functions.py' file in the same directory as this script to enable its functions for proper execution.
# 
# You're now ready to run the script for your XPCS analysis.
# 
# **Note:** If you encounter any issues, please ensure that all dependencies are correctly installed, and the 'XPCS_functions.py' file is in the same directory.
# 
# Make sure to follow these instructions for successful execution of your XPCS analysis script.

# In[ ]:


import os
import sys
from XPCS_functions import select_synchrotron, select_directory_or_files, generate_base_name
from XPCS_functions import ask_user_for_t_range, get_t_range_limits, ask_user_for_select_q
from XPCS_functions import initialize_error_and_success_counters, save_dataframe_to_tsv, save_t1t2_data
from XPCS_functions import process_sirius1_data, process_sirius2_data, process_sirius3_data, process_sirius3m_data, process_aps_data, process_esrf_data
import warnings
import tkinter as tk
from XPCS_functions import GraphWindow
from XPCS_functions import initialize_data_for_parameter_averages, initialize_plot, initialize_data_for_derived_params
import pandas as pd
import matplotlib.pyplot as plt
import re
from XPCS_functions import fit_single_exponential, fit_stretched_exponential, fit_cumulants, fit_and_plot_model
import math
from XPCS_functions import calculate_relaxation_and_diffusion
from XPCS_functions import add_data_to_table, write_dat_file
from XPCS_functions import update_parameter_tracking_r
from statistics import mean, stdev, StatisticsError
from XPCS_functions import plot_data_and_curves, plot_params, configure_subplot, calculate_text_position, calculate_average_parameter_values
from matplotlib.backends.backend_pdf import PdfPages
from XPCS_functions import print_summary, generate_fit_results_tables

# R2 Threshold
# This variable represents the minimum R-squared (R2) value required for a fit to be considered
# valid and included in the calculation of average parameter values. Fits with an R2 value
# below this threshold will be excluded from the averaging process.

R2_threshold = 0.9

#--------------------- Directory and File Handling ---------------------#

# Default directory (can be left empty)
directory = ''

# Get the list of .hdf5 files in the directory (if directory is defined)
if directory:
    hdf5_files = [file for file in os.listdir(directory) if file.endswith('.hdf5')]

# Get the selected synchrotron and data
selected_synchrotron, data_type = select_synchrotron()

# Exit if no synchrotron selected
if selected_synchrotron is None:
    print("User closed the window.")
    sys.exit()
    
# Call the select_directory_or_files function only if the directory is not defined
if not directory:
    directory, hdf5_files = select_directory_or_files()

# Exit if no .hdf5 files are found
if not hdf5_files:
    print("No .hdf5 files found in the directory.")
    sys.exit()
        
#--------------------- Initializations ---------------------#

# Initialize error and success counters using the dictionary
counters = initialize_error_and_success_counters()

# Initialize empty lists for table data
table_data = {
    "single": [],     # Empty list for single table data
    "stretched": [],  # Empty list for stretched table data
    "cumulants": []   # Empty list for cumulants table data
}

#--------------------- Flags and Control Variables ---------------------#
    
# Flag to indicate whether there are multiple files
#multiple_files = len(hdf5_files) > 1
#multiple_files = True

# Flag to indicate the first iteration of the loop
first_iteration = True

# Flag to control whether the loop should repeat
repite_loop = True #VER
    
# Variable to store the user's choice for defining the range
define_t_range = ask_user_for_t_range()

# User's choice for range definition in case of multiple files
#if multiple_files:
#    define_t_range = ask_user_for_t_range()

#--------------------- Suppress Warnings ---------------------#

# Suppress XDG_SESSION_TYPE warning
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Suppress all warnings
#warnings.simplefilter("ignore")

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


#####################################
##### Loop over each .hdf5 file #####
#####################################

for hdf5_file in hdf5_files:
    file_path = os.path.join(directory, hdf5_file)

    counters['hdf5_files'] += 1  # +++ Increment the hdf5_file counter +++

    # Generate the appropriate base name for output file naming based on selected synchrotron
    base_name = generate_base_name(selected_synchrotron, hdf5_file)
    
    #**************************************************************#
    # Obtain g2 vs. t data for each q value and create a DataFrame #
    #**************************************************************#
    
    # Process 'Sirius version 1' data and get the DataFrame
    if selected_synchrotron == 'Sirius 1':
        dataset_df = process_sirius1_data(file_path)
        
    # Process 'Sirius version 2' data and get the DataFrame
    elif selected_synchrotron == 'Sirius 2':
        dataset_df, t1t2_dfs = process_sirius2_data(file_path)
        
    # Process 'Sirius version 3' data and get the DataFrame
    elif selected_synchrotron == 'Sirius 3':
        dataset_df, t1t2_dfs = process_sirius3_data(file_path,data_type)

    # Process 'Sirius version 3m' data and get the DataFrame
    elif selected_synchrotron == 'Sirius 3 average':
        dataset_df = process_sirius3m_data(file_path,data_type)
    
    # Process 'APS' data and get the DataFrame
    elif selected_synchrotron == 'APS':
        dataset_df = process_aps_data(file_path)
        
    # Process 'ESRF' data and get the DataFrame
    elif selected_synchrotron == 'ESRF':
        dataset_df = process_esrf_data(file_path)
    
    # Check if the DataFrame is None, indicating a failure in data processing
    if dataset_df is None:
        sys.exit()        # Terminate the program to handle the error or exception
        
    # +++ Increment the q_values counter +++
    q_count = dataset_df.shape[1] - 1       # Subtract 1 to exclude the 't' column
    counters['q_values'] += q_count     
    
    # Define the range (slider shown on the first iteration if user chooses)
    if first_iteration:
    #if first_iteration and multiple_files:
        lower_limit, upper_limit = get_t_range_limits(dataset_df, define_t_range)
        first_iteration = False                  # Set the flag to False             
    
    #***** Save the data to a .dat file (CSV format with tab delimiter) *****#
            
    # Specify the output file path
    output_dataframe = os.path.join(directory, f"{base_name}_export_DataFrame.dat")

    warnings.filterwarnings("ignore")
    #warnings.filterwarnings("ignore", category=FutureWarning)
    
    # Sort the columns by 'q' values
    q_columns_df = sorted(
        [col for col in dataset_df.columns if re.match(r"g2\(q=\d+\.\d+\)", col)],
        key=lambda x: float(re.search(r"q=(\d+\.\d+)", x).group(1)))
    
    # Reorder the DataFrame columns, keeping '# t' at the beginning
    dataset_df = dataset_df[['# t'] + q_columns_df]

    # Save DataFrame to a tab-separated values (TSV) file.
    save_dataframe_to_tsv(dataset_df, output_dataframe)

    # Save t1-t2 DataFrame to a tab-separated values (TSV) file for Sirius 3 in t1-vs-t2 directory
    if selected_synchrotron == 'Sirius 2' or selected_synchrotron == 'Sirius 3':
        save_t1t2_data(t1t2_dfs, directory, base_name)

    #--------------------- Initializations ---------------------#
    
    # Initialize data structures for parameter averages
    parameter_averages_data = initialize_data_for_parameter_averages()
    
    # Call the function to initialize the plot
    fig1, (ax_single, ax_stretched, ax_cumulants, ax_Cq2_si, ax_Cq_si), fig2, (ax_Cq2_st, ax_Cq_st, ax_gammaq, ax_Cq2_cu, ax_Cq_cu, ax_PDIq), cmap, lines_labels_dict = initialize_plot(q_count)

    # Initialize empty lists for derived and qualified parameters
    derived_params, qualified_params = initialize_data_for_derived_params()
    
    #--------------------- --------------- ---------------------#
    
    #############################################################
    # If single file:
    if len(hdf5_files) == 1:
        
        # Ask the user if they want to select q values
        select_q_values = ask_user_for_select_q()
        
        if select_q_values:
            # Create a copy of the dataset
            original_dataset_df = dataset_df.copy()

            # Create the Tkinter window
            root = tk.Tk()

            # Create an instance of the GraphWindow class
            graph_window = GraphWindow(root, dataset_df, lower_limit, upper_limit, cmap)
            #graph_window = GraphWindow(root, dataset_df, cmap)

            # Start the Tkinter event loop
            root.mainloop()

            # Get the updated dataset after closing the window
            dataset_df = graph_window.select_values()

            # Close the Tkinter window
            root.quit()
            root.destroy()

            ## Save the original DataFrame to a tab-separated values (TSV) file.

            # Specify the output file path
            output_df_original = os.path.join(directory, f"{base_name}_export_DForiginal.dat")
            # Save DataFrame to a TSV file.
            save_dataframe_to_tsv(original_dataset_df, output_df_original)

    #############################################################
    
    ##################################
    ##### Loop over each q value #####
    ##################################
    
    for i, q_column in enumerate(dataset_df.columns[1:], 1):

        # Extract the q value from the column name
        match = re.search(r"q=(\d+\.\d+)", q_column)
        if match:
            q_value = float(match.group(1))
            
        # Get the 't' and 'g2' data for the current q value
        t = dataset_df['# t']
        g2 = dataset_df[q_column]

        # If the user has chosen to define the range of 't', apply the specified lower and upper limits
        if define_t_range:
            # Slice the values of 't' and 'g2' using the provided indices
            t = t.iloc[lower_limit:upper_limit + 1]               # Adding 1 to include the upper limit
            g2 = g2.iloc[lower_limit:upper_limit + 1]             # Adding 1 to include the upper limit

        # Create the output filename
        output_name = f"{base_name}_export_{chr(ord('a')+i-1)}.dat"

        # Generate the complete output file path
        output_path = os.path.join(directory, output_name)
        
        #------------------------------------------------------#
        #--------------------- Fit models ---------------------#
        #------------------------------------------------------#
        
        #---------- Fit the Single Exponential Model ----------#
        
        # Fit Single Exponential Model
        fit_params_single, fitted_curve_single, r2_single = fit_single_exponential(t, g2)
        
        # Add the filename to the list if fitting fails
        if math.isnan(r2_single): 
            counters['failure'] += 1        # +++ Increment the failure counter +++
            counters['failed_files'].append(f"{base_name} - q = {q_value} A^-1")

        # Calculate Relaxation time and Diffusion coefficient for Single Exponential
        relax_time_single, diffusion_coef_single = calculate_relaxation_and_diffusion(fit_params_single, q_value)
            
        #--------- Fit the Stretched Exponential model ---------#
        
        # Fit Stretched Exponential Model
        fit_params_stretched, fitted_curve_stretched, r2_stretched = fit_stretched_exponential(t, g2, fit_params_single)
        
        if math.isnan(r2_stretched):
            # Create the filename string
            failed_file = f"{base_name} - q = {q_value} A^-1"
            # Add the filename to the list if it's not already present
            if failed_file not in counters['failed_files']:
                counters['failure'] += 1       # +++ Increment the failure counter +++
                counters['failed_files'].append(failed_file)

        # Calculate Relaxation time and Diffusion coefficient for Stretched Exponential
        relax_time_stretched, diffusion_coef_stretched = calculate_relaxation_and_diffusion(fit_params_stretched, q_value)
        
        #--------- Analysis by the method of Cumulants ---------#
        
        # Fit Cumulants Model
        fit_params_cumulants, fitted_curve_cumulants, r2_cumulants = fit_cumulants(t, g2, fit_params_single)
        
        if math.isnan(r2_cumulants):
            # Create the filename string
            failed_file = f"{base_name} - q = {q_value} A^-1"
            # Add the filename to the list if it's not already present
            if failed_file not in counters['failed_files']:
                counters['failure'] += 1       # +++ Increment the failure counter +++
                counters['failed_files'].append(failed_file)

        # Calculate Relaxation time and Diffusion coefficient for Cumulants
        relax_time_cumulants, diffusion_coef_cumulants = calculate_relaxation_and_diffusion(fit_params_cumulants, q_value)

        #******************************************************#
        #** Append data to the table and save to a .dat file **#
        #******************************************************#

        # for Single Exponential
        A_single = fit_params_single[0]
        B_single = fit_params_single[1]
        C_single = fit_params_single[2]
        # Append data to the table for Single Exponential
        add_data_to_table(table_data["single"], base_name, q_value, [A_single, B_single, C_single], relax_time_single, diffusion_coef_single, r2_single)

        # for Stretched Exponential
        A_stretched = fit_params_stretched[0]
        B_stretched = fit_params_stretched[1]
        C_stretched = fit_params_stretched[2]
        gamma_stretched = fit_params_stretched[3]
        # Append data to the table for Stretched Exponential
        add_data_to_table(table_data["stretched"], base_name, q_value, [A_stretched, B_stretched, C_stretched, gamma_stretched], relax_time_stretched, diffusion_coef_stretched, r2_stretched)
            
        # for Cumulants Model
        A_cumulants = fit_params_cumulants[0]
        B_cumulants = fit_params_cumulants[1]
        C1_cumulants = fit_params_cumulants[2]
        C2_cumulants = fit_params_cumulants[3]
        PDI_cumulants = C2_cumulants / C1_cumulants**2
        # Append data to the table for Cumulants Model
        add_data_to_table(table_data["cumulants"], base_name, q_value, [A_cumulants, B_cumulants, C1_cumulants, PDI_cumulants], relax_time_cumulants, diffusion_coef_cumulants, r2_cumulants)
            
        # Write data to .dat file for Single Exponential, Stretched Exponential and Cumulants
        write_dat_file(output_path, q_value, t, g2, A_single, B_single, C_single, A_stretched, B_stretched, C_stretched, gamma_stretched,
            r2_single, r2_stretched, relax_time_single, relax_time_stretched, diffusion_coef_single, diffusion_coef_stretched,
            A_cumulants, B_cumulants, C1_cumulants, C2_cumulants, r2_cumulants, relax_time_cumulants, diffusion_coef_cumulants)

        #********************************************************#
        #***** Add the data for derived params in the lists *****#
        #********************************************************#
        
        derived_params["q_value"].append(q_value)
        derived_params["q_square"].append(q_value**2)
        derived_params["relax_rate_single"].append(C_single)
        derived_params["relax_rate_stretched"].append(C_stretched)
        derived_params["gamma"].append(gamma_stretched)
        derived_params["relax_rate_cumulants"].append(C1_cumulants)
        derived_params["PDI"].append(PDI_cumulants)
        
        #********************************************************#
        
        ########################################################
        ### Plot the experimental data and the fitted curves ###
        ########################################################

        color = cmap(i-1)  # Get color based on index
        #print(f"i: {i}, color: {color}")
            
        # for Single Exponential
        plot_data_and_curves(ax_single, t, g2, A_single, B_single, fitted_curve_single, q_value, r2_single, lines_labels_dict, color, linestyle='-', model_type = "Single")

        # for Stretched Exponential
        plot_data_and_curves(ax_stretched, t, g2, A_stretched, B_stretched, fitted_curve_stretched, q_value, r2_stretched, lines_labels_dict, color, linestyle='--', model_type = "Stretched")
            
        # for Method of Cumulants
        plot_data_and_curves(ax_cumulants, t, g2, A_cumulants, B_cumulants, fitted_curve_cumulants, q_value, r2_cumulants, lines_labels_dict, color, linestyle='dotted', model_type = "Cumulants")
            
        #****************************************************************#
        # Calculation of the average parameters of the fits with R > 0.9 #
        #****************************************************************#
            
        # Check if r2_single is not NaN and if it's greater than R2_threshold before updating parameter tracking.
        if not math.isnan(r2_single) and r2_single > R2_threshold:
            update_parameter_tracking_r(parameter_averages_data, 'single', r2_single, diffusion_coef_single)
            qualified_params["q_value_single"].append(q_value)
            qualified_params["relax_rate_single"].append(C_single)
            qualified_params["color_single"].append(color)
        
        # Check if r2_stretched is not NaN and if it's greater than R2_threshold before updating parameter tracking.
        if not math.isnan(r2_stretched) and r2_stretched > R2_threshold:
            update_parameter_tracking_r(parameter_averages_data, 'stretched', r2_stretched, diffusion_coef_stretched, gamma_stretched)
            qualified_params["q_value_stretched"].append(q_value)
            qualified_params["relax_rate_stretched"].append(C_stretched)
            qualified_params["color_stretched"].append(color)
            qualified_params["gamma"].append(gamma_stretched)

        # Check if r2_cumulants is not NaN and if it's greater than R2_threshold before updating parameter tracking.
        if not math.isnan(r2_cumulants) and r2_cumulants > R2_threshold:
            update_parameter_tracking_r(parameter_averages_data, 'cumulants', r2_cumulants, diffusion_coef_cumulants, PDI_cumulants)         
            qualified_params["q_value_cumulants"].append(q_value)
            qualified_params["relax_rate_cumulants"].append(C1_cumulants)
            qualified_params["color_cumulants"].append(color)
            qualified_params["PDI"].append(PDI_cumulants)
            
        #****************************************************************#                        
            
        counters['success'] += 1  # +++ Increment the hdf5_file counter +++
        
    ########################################################################
    ##################### Fit and plot derived params  #####################
    ########################################################################
    
    # Disable the warning
    warnings.filterwarnings("ignore", category=UserWarning, message="No handles with labels found to put in legend.")
    
    #--------------------------------------------------#
    #----------- Fit the curve C = D * q**2 -----------#
    #--------------------------------------------------#
    
    #--------- Fit C vs q^2 ---------#  #----Linear----#
    
    # Call the function to perform the fit and generate the plot for each model
    D_single_linear, n_single_linear, r_single_linear, std_err_single_linear = fit_and_plot_model(qualified_params["q_value_single"], qualified_params["relax_rate_single"], ax_Cq2_si, qualified_params["color_single"], "Single - Linear", model_type='linear')
    D_stretched_linear, n_stretched_linear, r_stretched_linear, std_err_stretched_linear = fit_and_plot_model(qualified_params["q_value_stretched"], qualified_params["relax_rate_stretched"], ax_Cq2_st, qualified_params["color_stretched"], 'Stretched - Linear', model_type='linear')
    D_cumulants_linear, n_cumulants_linear, r_cumulants_linear, std_err_cumulants_linear = fit_and_plot_model(qualified_params["q_value_cumulants"], qualified_params["relax_rate_cumulants"], ax_Cq2_cu, qualified_params["color_cumulants"], 'Cumulants - Linear', model_type='linear')
    
    # Set labels and annotations for all three models
    for ax, D_values, pearson_r_values, std_err_values in [(ax_Cq2_si, D_single_linear, r_single_linear, std_err_single_linear),
                                                           (ax_Cq2_st, D_stretched_linear, r_stretched_linear, std_err_stretched_linear),
                                                           (ax_Cq2_cu, D_cumulants_linear, r_cumulants_linear, std_err_cumulants_linear)]:

        # Set labels
        ax.set_xlabel("$q^{2} (\AA^{-2}$)", fontsize=14)
        ax.set_ylabel("Relaxation Rate ($s^{-1}$)", fontsize=14)
        ax.set_title("Relax rate vs $q^{2}$ (Linear Fit)", fontsize=16)
        #ax.legend()

        # Annotate the graph with slopes (D) and Pearson correlation coefficients (pearson_r)
        slope_annotation = f"Slope (D) = {D_values:.2e}"
        r_annotation = f"Pearson r = {pearson_r_values:.4f}"

        # Annotate the graph with diffusion coefficient and standard deviation
        diff_coef = D_values / 1e8
        std_err_diff_coef = std_err_values / 1e8
        diff_coef_annotation = fr"Diffusion Coefficient ($\mu m^2/s$) = {diff_coef:.4f} ({std_err_diff_coef:.4f})"

        # Get the coordinates of the graph label
        label_x, label_y = 0.02, ax.get_ylim()[1]

        # Add the slope_annotation and r_annotation in the lower-right corner
        ax.annotate(slope_annotation, (0.98, 0.05), xycoords='axes fraction', color="black", fontsize=12, ha='right', va='bottom')
        ax.annotate(r_annotation, (0.98, 0.10), xycoords='axes fraction', color="black", fontsize=12, ha='right', va='bottom')

        # Add the diff_coef_annotation in the upper-left corner
        ax.annotate(diff_coef_annotation, (0.02, 0.95), xycoords='axes fraction', color="black", fontsize=12, ha='left', va='top')
        
    #---------- Fit C vs q ----------# #--Exponential--#
    
    # Call the function to perform the fit and generate the plot for each model
    D_single_exponential, n_single_exponential, r_single_exponential, std_err_single_exponential = fit_and_plot_model(qualified_params["q_value_single"], qualified_params["relax_rate_single"], ax_Cq_si, qualified_params["color_single"], "Single - Exponential", model_type='exponential')
    D_stretched_exponential, n_stretched_exponential, r_stretched_exponential, std_err_stretched_exponential = fit_and_plot_model(qualified_params["q_value_stretched"], qualified_params["relax_rate_stretched"], ax_Cq_st, qualified_params["color_stretched"], 'Stretched - Exponential', model_type='exponential')
    D_cumulants_exponential, n_cumulants_exponential, r_cumulants_exponential, std_err_cumulants_exponential = fit_and_plot_model(qualified_params["q_value_cumulants"], qualified_params["relax_rate_cumulants"], ax_Cq_cu, qualified_params["color_cumulants"], 'Cumulants - Exponential', model_type='exponential')
    
    # Set labels and annotations for all three models
    for ax, D_values, n_values, r2_values in [(ax_Cq_si, D_single_exponential, n_single_exponential, r_single_exponential),
                                               (ax_Cq_st, D_stretched_exponential, n_stretched_exponential, r_stretched_exponential),
                                               (ax_Cq_cu, D_cumulants_exponential, n_cumulants_exponential, r_cumulants_exponential)]:
        # Set labels
        ax.set_xlabel("q ($\AA^{-1}$)", fontsize=14)
        ax.set_ylabel("Relaxation Rate ($s^{-1}$)", fontsize=14)
        ax.set_title("Relax rate vs q (Exponential Fit)", fontsize=16)
        #ax.legend()

        # Annotate the graph with rate constant (D), exponent (n), and Coefficient of determination (R^2)
        rate_constant_annotation = f"Rate constant = {D_values:.2e}"
        n_annotation = f"Exponent = {n_values:.2f}"
        r2_annotation = f"R-squared = {r2_values:.4f}"

        # Get the coordinates of the graph label
        label_x, label_y = 0.02, ax.get_ylim()[1]

        # Add the label in the upper-left corner without a border
        ax.annotate(rate_constant_annotation, (0.02, 0.95), xycoords='axes fraction', color="black", fontsize=12)
        ax.annotate(n_annotation, (0.02, 0.90), xycoords='axes fraction', color="black", fontsize=12)
        ax.annotate(r2_annotation, (0.02, 0.85), xycoords='axes fraction', color="black", fontsize=12)

    ################################################################
    ##################### Plot derived params  #####################
    ################################################################
    
    #--------------- Plot Gamma C vs q ---------------#
    
    # Call the plot_params function
    warnings.filterwarnings("ignore")
    plot_params(derived_params["q_value"], derived_params["gamma"], ax_gammaq, cmap, "gamma")

    # Add labels and legend
    ax_gammaq.set_xlabel("q ($\AA^{-1}$)", fontsize=14)
    ax_gammaq.set_ylabel("Gamma", fontsize=14)
    ax_gammaq.set_title("Gamma vs q", fontsize=16)
    #ax_gammaq.legend()
    
    # Calculate the mean and standard deviation
    try:
        average_gamma = mean(qualified_params["gamma"])
        stdev_gamma = stdev(qualified_params["gamma"])
    
        # Add the label with the mean and standard deviation of gamma in the upper-left corner without a border
        ax_gammaq.annotate(f"Av. Gamma = {average_gamma:.2f} ({stdev_gamma:.2f})", (0.02, 0.95), xycoords='axes fraction', color="black", fontsize=12)
    
    except StatisticsError:
        pass
    
    #---------------- Plot PDI C vs q ----------------#
    
    # Call the plot_params function
    warnings.filterwarnings("ignore")
    plot_params(derived_params["q_value"], derived_params["PDI"], ax_PDIq, cmap, "PDI")

    # Add labels and legend
    ax_PDIq.set_xlabel("q ($\AA^{-1}$)", fontsize=14)
    ax_PDIq.set_ylabel("PDI", fontsize=14)
    ax_PDIq.set_title("PDI vs q", fontsize=16)
    #ax_PDIq.legend()
    
    # Calculate the mean and standard deviation
    try:
        average_PDI = mean(qualified_params["PDI"])
        stdev_PDI = stdev(qualified_params["PDI"])
    
        # Add the label with the mean and standard deviation of PDI in the upper-left corner without a border
        ax_PDIq.annotate(f"Av. PDI = {average_PDI:.2f} ({stdev_PDI:.2f})", (0.02, 0.95), xycoords='axes fraction', color="black", fontsize=12)
    
    except StatisticsError:
        pass
    
    ###############################################
    ### Configure the subplots for the 3 models ###
    ###############################################
    
    # Calculate the text position on the plot
    diff_coef_pos = calculate_text_position(q_count)
    
    # Configure subplot for Single Exponential
    configure_subplot(ax_single, "Single Exponential", "Delay Time (s)", r"$(g_2 - \mathrm{base}) / \beta$",
                      lines_labels_dict["exp_lines_single"],
                      lines_labels_dict["exp_labels_single"],
                      lines_labels_dict["fit_lines_single"],
                      lines_labels_dict["fit_labels_single"])
    
    # Calculate average parameter values for the 'single' model if R count is not zero
    if parameter_averages_data['single']['R count'] > 0:
        average_single_parameters = calculate_average_parameter_values(parameter_averages_data, 'single') 
        
        # Add average parameter values to Single Exponential subplot
        ax_single.text(1, diff_coef_pos, fr"Av. Diff Coef ($\mu m^{2}/s$): {average_single_parameters['Diff_coef av']:.4f} ({average_single_parameters['Diff_coef std']:.4f})",
                          transform=ax_single.transAxes, va='top', ha='right', fontsize=12)        

    # Configure subplot for Stretched Exponential
    configure_subplot(ax_stretched, "Stretched Exponential", "Delay Time (s)", r"$(g_2 - \mathrm{base}) / \beta$",
                      lines_labels_dict["exp_lines_stretched"],
                      lines_labels_dict["exp_labels_stretched"],
                      lines_labels_dict["fit_lines_stretched"],
                      lines_labels_dict["fit_labels_stretched"])
    
    # Calculate average parameter values for the 'stretched' model if R count is not zero
    if parameter_averages_data['stretched']['R count'] > 0:
        average_stretched_parameters = calculate_average_parameter_values(parameter_averages_data, 'stretched')
        
        # Add average parameter values to Stretched Exponential subplot       
        ax_stretched.text(1, diff_coef_pos, fr"Av. Diff Coef ($\mu m^{2}/s$): {average_stretched_parameters['Diff_coef av']:.4f} ({average_stretched_parameters['Diff_coef std']:.4f})",
                          transform=ax_stretched.transAxes, va='top', ha='right', fontsize=12)
        ax_stretched.text(1, diff_coef_pos-0.045, f"Av. Gamma: {average_stretched_parameters['Gamma av']:.4f} ({average_stretched_parameters['Gamma std']:.4f})", 
                          transform=ax_stretched.transAxes, va='top', ha='right', fontsize=12)

    # Configure subplot for Cumulants model
    configure_subplot(ax_cumulants, "Cumulants", "Delay Time (s)", r"$(g_2 - \mathrm{base}) / \beta$",
                      lines_labels_dict["exp_lines_cumulants"],
                      lines_labels_dict["exp_labels_cumulants"],
                      lines_labels_dict["fit_lines_cumulants"],
                      lines_labels_dict["fit_labels_cumulants"])
    
    # Calculate average parameter values for the 'cumulants' model if R count is not zero
    if parameter_averages_data['cumulants']['R count'] > 0:
        average_cumulants_parameters = calculate_average_parameter_values(parameter_averages_data, 'cumulants')
        
        # Add average parameter values to Cumulants subplot
        ax_cumulants.text(1, diff_coef_pos, fr"Av. Diff Coef ($\mu m^{2}/s$): {average_cumulants_parameters['Diff_coef av']:.4f} ({average_cumulants_parameters['Diff_coef std']:.4f})", 
                          transform=ax_cumulants.transAxes, va='top', ha='right', fontsize=12)
        ax_cumulants.text(1, diff_coef_pos-0.045, f"Av. PDI: {average_cumulants_parameters['PDI av']:.4f} ({average_cumulants_parameters['PDI std']:.4f})", 
                          transform=ax_cumulants.transAxes, va='top', ha='right', fontsize=12)

    ###############################################
    
    # Save the figures to a PDF file with two pages
    pdf_name = f"{base_name}.pdf"
    output_pdf_path = os.path.join(directory, pdf_name)

    with PdfPages(output_pdf_path) as pdf:
        # Save the first figure and its subplots
        pdf.savefig(fig1)

        # Save the second figure and its subplots
        pdf.savefig(fig2)

    # Close the figure to free up memory
    plt.close(fig1)
    plt.close(fig2)
    
    # Sort the columns of combined_df based on q values
    dataset_df = dataset_df[['# t'] + sorted(dataset_df.columns[1:], key=lambda col_name: float(re.search(r'q=([0-9.]+)', col_name).group(1)))]

    # Save the DataFrame to a .dat file
    dataset_df.to_csv(output_dataframe, sep='\t', index=False)

plt.close()
plt.close()
    
# Print the counts, filed files and invalid files
print_summary(counters)

# Generate the new .dat files with the fit results tables for each model
generate_fit_results_tables(directory, table_data["single"], table_data["stretched"], table_data["cumulants"])

