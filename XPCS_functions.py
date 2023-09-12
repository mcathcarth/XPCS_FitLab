import tkinter as tk
from qtpy.QtWidgets import QApplication, QFileDialog
import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
import warnings
from scipy.optimize import curve_fit
import os

# Function to select synchrotron
def select_synchrotron():
    """
    Displays a GUI window to select a synchrotron and returns the selected synchrotron.
    
    Returns:
        str: The selected synchrotron.
    """

    # Initialize the synchrotron variable
    synchrotron = None

    # Define the nested functions to set the synchrotron:
    def set_sirius():
        nonlocal synchrotron         # Access the outer variable
        synchrotron = 'Sirius'       # Set the synchrotron to 'Sirius'
        root.destroy()               # Close the GUI window

    def set_APS():
        nonlocal synchrotron
        synchrotron = 'APS'
        root.destroy()

    # Set synchrotron to None when the window is closed
    def on_closing():
        nonlocal synchrotron
        synchrotron = None  
        root.destroy()

    # Create the main GUI window
    root = tk.Tk()
    root.title("Select Synchrotron")    # Set the title of the window

    # Create a label widget to display the instruction
    label = tk.Label(root, text="Select the synchrotron:")
    label.pack(pady=10)                 # Add some padding

    # Create the buttons to select the synchrotron 
    sirius_button = tk.Button(root, text="Sirius", command=set_sirius)
    sirius_button.pack(pady=5)
    
    APS_button = tk.Button(root, text="APS", command=set_APS)
    APS_button.pack(pady=5)

    # Bind the closing of the window to the on_closing function
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start the main GUI event loop
    root.mainloop()
    
    return synchrotron

# Function to select a directory
def select_directory():
    """
    Displays a GUI window to select a directory and returns the selected directory path.
    
    Returns:
        str: The selected directory path.
    """
    
    # Initialize the QtPy application
    app = QApplication([])

    # Prompt for directory if not defined
    directory = QFileDialog.getExistingDirectory(None, "Select directory")

    # Exit if no directory is selected
    if not directory:
        print("No directory selected.")
        sys.exit()

    # Return the selected directory
    return directory

# Function to initialize error and success counters
def initialize_error_and_success_counters():
    """
    Initializes counters and lists for error and success tracking.
    
    Returns:
        dict: A dictionary containing counters and lists for error and success tracking.
    """
    counters = {
        'hdf5_files': 0,
        'multi_tau': 0,
        'invalid_data': 0,
        'success': 0,
        'failure': 0,
        'invalid_files': [],
        'failed_files': []
    }
    
    return counters

# Function to generate the base name
def generate_base_name(selected_synchrotron, hdf5_file):
    """
    Generates a base name for data processing based on the selected synchrotron.

    Parameters:
        selected_synchrotron (str): The selected synchrotron name ('Sirius' or 'APS').
        hdf5_file (str): The HDF5 file name.

    Returns:
        str: The generated base name.
    """
    if selected_synchrotron == 'Sirius':
        base_name = hdf5_file.replace('_saxs_0000_RESULTS.hdf5', '')
    elif selected_synchrotron == 'APS':
        base_name = hdf5_file.replace('.hdf5', '')
    else:
        base_name = hdf5_file

    return base_name

# Function to process 'Multi-tau' data for Sirius
def process_multi_tau_sirius(hdf):
    """
    Process the 'Multi-tau' data for Sirius.

    Parameters:
        hdf (h5py.File): The HDF5 file object.

    Returns:
        list: List of dataset keys within 'Multi-tau', 
        object: 'Multi-tau' HDF5 group object.
        Returns None if processing is skipped.
    """
    try:
        # Check if 'Multi-tau' key exists in HDF5 file
        multi_tau = hdf['Multi-tau']
        
        # Read the keys of the datasets within Multi-tau
        dataset_keys = list(multi_tau.keys())
    except KeyError:
        print(f"Skipping file: {hdf5_file}. 'Multi-tau' key not found.")
        return None
    
    return dataset_keys, multi_tau

# Function to process 'Multi-tau' data for APS
def process_multi_tau_aps(hdf):
    """
    Process the 'Multi-tau' data for APS.

    Parameters:
        hdf (h5py.File): The HDF5 file object.

    Returns:
        list: List of dataset keys within 'Multi-tau', or None if processing is skipped.
    """
    return None

# Function to initialize a plot
def initialize_plot():
    """
    Initialize a matplotlib plot with specified LaTeX font and other settings.
    
    Returns:
        tuple: A tuple containing figure, axes, color map, and a dictionary of lists for organizing data.
               The dictionary keys include:
               - 'exp_lines_single', 'exp_labels_single', 'fit_lines_single', 'fit_labels_single'
               - 'exp_lines_stretched', 'exp_labels_stretched', 'fit_lines_stretched', 'fit_labels_stretched'
    """
    
    # Configure LaTeX font with Cambria Math
    #plt.rcParams["text.usetex"] = True
    #plt.rcParams["text.latex.preamble"] = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{fontspec} \setmainfont{Cambria Math}'
    
    # Create a new figure for each HDF5 file
    fig, (ax_single, ax_stretched) = plt.subplots(1, 2, figsize=(20, 8))

    # Initialize a color map for different q values
    cmap = plt.get_cmap('tab10')

    # Initialize a dictionary to store lines and labels for legend items
    lines_labels_dict = {
        "exp_lines_single": [],    # Lines for experimental data for Single Exponential
        "exp_labels_single": [],   # Labels for experimental data for Single Exponential
        "fit_lines_single": [],    # Lines for fitted curves for Single Exponential
        "fit_labels_single": [],   # Labels for fitted curves for Single Exponential
        "exp_lines_stretched": [],  # Lines for experimental data for Stretched Exponential
        "exp_labels_stretched": [], # Labels for experimental data for Stretched Exponential
        "fit_lines_stretched": [],  # Lines for fitted curves for Stretched Exponential
        "fit_labels_stretched": []  # Labels for fitted curves for Stretched Exponential
    }
    
    return fig, ax_single, ax_stretched, cmap, lines_labels_dict

# Function to fit a given model to the data and return parameters, fitted curve, and R2 score
def fit_model(t, g2, model_func, initial_params):
    """
    Fit a given model to the data and return parameters, fitted curve, and R2 score.

    Parameters:
        t (array-like): The time values.
        g2 (array-like): The experimental data.
        model_func (function): The model function to fit.
        initial_params (list): Initial guess for the model parameters.

    Returns:
        tuple: Fit parameters, fitted curve, and R2 score.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore warnings
            fit_params, _ = curve_fit(model_func, t, g2, p0=initial_params)
        fitted_curve = model_func(t, *fit_params)
        r2 = r2_score(g2, fitted_curve)
        return fit_params, fitted_curve, r2
    except (RuntimeError, ValueError):
        return np.full(len(initial_params), np.nan), np.full(len(t), np.nan), np.nan

# Single Exponential Model function
def single_exponential(t, A, B, C):
    """
    Single Exponential model function.

    Parameters:
        t (array-like): The time values.
        A (float): Parameter A.
        B (float): Parameter B.
        C (float): Parameter C.

    Returns:
        array-like: The fitted curve.
    """
    return A + B * np.exp(-2 * C * t)

# Stretched Exponential Model function
def stretched_exponential(t, A, B, C, gamma):
    """
    Stretched Exponential model function.

    Parameters:
        t (array-like): The time values.
        A (float): Parameter A.
        B (float): Parameter B.
        C (float): Parameter C.
        gamma (float): Parameter gamma.

    Returns:
        array-like: The fitted curve.
    """
    return A + B * np.exp(-2 * C * t) ** gamma

# Function to calculate relaxation time and diffusion coefficient
def calculate_relaxation_and_diffusion(params, q_value):
    """
    Calculate relaxation time and diffusion coefficient.

    Parameters:
        params (array-like): Fit parameters from the model.
        q_value (float): The q value.

    Returns:
        float: Relaxation time.
        float: Diffusion coefficient.
    """
    try:
        C = params[2]
        relax_time = 1 / C                                      # 1/C
        diffusion_coef = (C / (q_value ** 2)) * 0.00000001      # C/q^2
    except (IndexError, ZeroDivisionError):
        relax_time = np.nan
        diffusion_coef = np.nan
    return relax_time, diffusion_coef   

# Function to plot experimental data and fitted curves
def plot_data_and_curves(ax, t, g2, fitted_curve, q_value, r2_value, lines_labels_dict, color, linestyle, model_type):
    """
    Add experimental data and fitted curves to a plot.

    Args:
        ax (matplotlib.axes.Axes): The Axes object to which data will be added.
        t (numpy.ndarray): Array of delay times.
        g2 (numpy.ndarray): Array of experimental g2 values.
        fitted_curve (numpy.ndarray): Fitted curve.
        q_value (float): Value of q.
        r2_value (float): R-squared value.
        lines_labels_dict (dict): Dictionary containing lists for organizing data.
        color (str): The color for the plot elements.
        linestyle (str, optional): The linestyle for the fitted curve. Defaults to '-'.
        model_type (str): The type of model, either "Single" or "Stretched".

    Returns:
        None
    """
    
    # Plot experimental data points
    line_exp = ax.semilogx(t, (g2 - 1), 'o', color=color)[0]

    # Plot fitted curve
    line_fit = ax.semilogx(t, (fitted_curve - 1), color=color, linestyle=linestyle)[0]

    # Add labels to the legend lists based on the model type
    lines_labels_dict[f"exp_lines_{model_type.lower()}"].append(line_exp)
    lines_labels_dict[f"exp_labels_{model_type.lower()}"].append(f"q = {q_value:.6f}")
    lines_labels_dict[f"fit_lines_{model_type.lower()}"].append(line_fit)
    lines_labels_dict[f"fit_labels_{model_type.lower()}"].append(f"R2: {r2_value:.2f}")

# Function to add the data to the table
def add_data_to_table(table_data, base_name, q_value, params, relax_time, diffusion_coef, r2):
    """
    Appends data to a table.

    Parameters:
        table_data (list): The list containing the table data.
        base_name (str): The base name.
        q_value (float): The q-value.
        params (list): List of parameters to be added to the table.
        relax_time (float): The relaxation time.
        diffusion_coef (float): The diffusion coefficient.
        r2 (float): The R2 score.
    """
    table_row = [base_name, q_value, *params, relax_time, diffusion_coef, r2]
    table_data.append(table_row)

# Function to write data to .dat file
def write_dat_file(output_path, q_value, t, g2, A, B, C, A_stretched, B_stretched, C_stretched, gamma,
                   r2_single, r2_stretched, relax_time_single, relax_time_stretched, diffusion_coef_single, diffusion_coef_stretched):
    """
    Write data and comments to a .dat file.

    Parameters:
        output_path (str): The path to the output .dat file.
        q_value (float): The q-value associated with the data.
        t (array-like): Time values.
        g2 (array-like): Experimental data.
        A, B, C, A_stretched, B_stretched, C_stretched, gamma (float): Fit parameters.
        r2_single, r2_stretched (float): R2 scores.
        relax_time_single, relax_time_stretched (float): Relaxation times.
        diffusion_coef_single, diffusion_coef_stretched (float): Diffusion coefficients.
    """
    with open(output_path, 'w') as dat_file:
        dat_file.write(f"q = {q_value} A^-1\n")
        dat_file.write("### Fit model: Single Exponential ###\n")
        dat_file.write(f"#Fit parameters: baseline = {A}, beta = {B}, relax rate (1/s) = {C}\n")
        dat_file.write(f"#R2: {r2_single}\n")
        dat_file.write(f"#Derived parameters: Relax. time (s) = {relax_time_single}, Diffusion coef (u2/s) = {diffusion_coef_single}\n")
        dat_file.write("### Fit model: Stretched Exponential ###\n")
        dat_file.write(f"#Fit parameters: baseline = {A_stretched}, beta = {B_stretched}, relax rate (1/s) = {C_stretched}, gamma = {gamma}\n")
        dat_file.write(f"#R2: {r2_stretched}\n")
        dat_file.write(f"#Derived parameters: Relax. time (s) = {relax_time_stretched}, Diffusion coef (u2/s) = {diffusion_coef_stretched}\n")
        dat_file.write("#delay time (s)\tg2\tstd\tFitted g2 Single\tFitted g2 Stretched\n")

        fitted_curve_single = A + B * np.exp(-2 * C * t)
        fitted_curve_stretched = A_stretched + B_stretched * np.exp(-2 * C_stretched * t) ** gamma

        for i in range(len(t)):
            row_str = f"{t[i]}\t{g2[i]}\t{np.std(g2)}\t{fitted_curve_single[i]}\t{fitted_curve_stretched[i]}\n"
            dat_file.write(row_str)

# Configure subplot with title, labels, experimental data legend, and fitted curves legend
def configure_subplot(ax, title, xlabel, ylabel, exp_lines, exp_labels, fit_lines, fit_labels):
    """
    Configures a subplot with title, labels, experimental data legend, and fitted curves legend.

    Parameters:
        ax (matplotlib.axes._subplots.AxesSubplot): The subplot to configure.
        title (str): The title of the subplot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        exp_lines (list): List of lines for experimental data.
        exp_labels (list): List of labels for experimental data.
        fit_lines (list): List of lines for fitted curves.
        fit_labels (list): List of labels for fitted curves.
    """
    # Set the title and labels for the subplot
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Create the legend for experimental data
    legend_exp = ax.legend(exp_lines, exp_labels, loc='upper right', bbox_to_anchor=(0.87, 1), borderaxespad=0)
    ax.add_artist(legend_exp)

    # Create the legend for fitted curves
    legend_fit = ax.legend(fit_lines, fit_labels, loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0)
    ax.add_artist(legend_fit)

# Function to print error and success counters
def print_summary(counters):
    """
    Print a summary of processing results.

    Parameters:
        counters (dict): A dictionary containing counters and lists for error and success tracking.
    """
    print(f"Total HDF5 files: {counters['hdf5_files']}")
    print(f"Total x files: {counters['multi_tau']}")
    print(f"Total invalid files: {counters['invalid_data']}")

    # Print the base_names of invalid files
    if counters['invalid_data'] != 0:
        print("Invalid files:")
        for file_name in counters['invalid_files']:
            print(file_name)
    
    print(f"Successful fits: {counters['success']}")
    print(f"Failed fits: {counters['failure']}")

    # Print the base_names of failed files
    if counters['failure'] != 0:
        print("Failed fits files:")
        for file_name in counters['failed_files']:
            print(file_name)

# Function to generate fit results tables
def generate_fit_results_tables(directory, table_data_single, table_data_stretched):
    """
    Generate .dat files with fit results tables for Single Exponential and Stretched Exponential models.

    Parameters:
        directory (str): Directory where the .dat files will be saved.
        table_data_single (list): List of lists containing data for Single Exponential table.
        table_data_stretched (list): List of lists containing data for Stretched Exponential table.
    """
    table_output_path_single = os.path.join(directory, "fit_results_single.dat")
    table_output_path_stretched = os.path.join(directory, "fit_results_stretched.dat")

    with open(table_output_path_single, 'w') as table_file_single, open(table_output_path_stretched, 'w') as table_file_stretched:
        # Write header for Single Exponential
        header_single = "{:^26} {:^13} {:^23} {:^24} {:^24} {:^26} {:^25} {:^24}\n".format(
            "Filename", "q (A^-1)", "Baseline", "beta", "Relax. time (s)", "Relax. rate (s-1)", "Diff. coefficient (u2/s)", "R2"
        )
        table_file_single.write(header_single)

        # Write header for Stretched Exponential
        header_stretched = "{:^26} {:^13} {:^23} {:^24} {:^24} {:^24} {:^26} {:^25} {:^24}\n".format(
            "Filename", "q (A^-1)", "Baseline", "beta", "Relax. rate (s-1)", "Gamma", "Relax. time (s)", "Diff. coefficient (u2/s)", "R2"
        )
        table_file_stretched.write(header_stretched)

        # Write table data for Single Exponential
        for row_single in table_data_single:
            row_str_single = "{:<26} {:<13} {:<23} {:<24} {:<24} {:<26} {:<25} {:<24}\n".format(*row_single)
            table_file_single.write(row_str_single)

        # Write table data for Stretched Exponential
        for row_stretched in table_data_stretched:
            row_str_stretched = "{:<26} {:<13} {:<23} {:<24} {:<24} {:<24} {:<26} {:<25} {:<24}\n".format(*row_stretched)
            table_file_stretched.write(row_str_stretched)

        # Write table data for Single Exponential
       # for row_single in table_data_single:
        #    row_str_single = "\t".join(str(item) for item in row_single)
         #   table_file_single.write(row_str_single + "\n")
        
        # Write table data for Stretched Exponential
      #  for row_stretched in table_data_stretched:
       #     row_str_stretched = "\t".join(str(item) for item in row_stretched)
        #    table_file_stretched.write(row_str_stretched + "\n")

    print("Fit results tables generated successfully.")


# Function to generate fit results tables
def generate_fit_results_tables2(directory, table_data_single, table_data_stretched):
    """
    Generate .dat files with fit results tables for Single Exponential and Stretched Exponential models.

    Parameters:
        directory (str): Directory where the .dat files will be saved.
        table_data_single (list): List of lists containing data for Single Exponential table.
        table_data_stretched (list): List of lists containing data for Stretched Exponential table.
    """
    table_output_path_single = os.path.join(directory, "fit_results_single.dat")
    table_output_path_stretched = os.path.join(directory, "fit_results_stretched.dat")

    with open(table_output_path_single, 'w') as table_file_single, open(table_output_path_stretched, 'w') as table_file_stretched:
        # Write header for Single Exponential
        header_single = "Filename\tq (A^-1)\tBaseline\tbeta\tRelax. time (s)\tRelax. rate (s-1)\tDiffusion coefficient (u2/s)\tR2\n"
        table_file_single.write(header_single)

        # Write header for Stretched Exponential
        header_stretched = "Filename\tq (A^-1)\tBaseline\tbeta\tRelax. rate (s-1)\t Gamma \tRelax. time (s)\tDiffusion coefficient (u2/s)\tR2\n"
        table_file_stretched.write(header_stretched)

        # Write table data for Single Exponential
        for row_single in table_data_single:
            row_str_single = '\t'.join(str(item) for item in row_single)
            table_file_single.write(f"{row_str_single}\n")
        
        # Write table data for Stretched Exponential
        for row_stretched in table_data_stretched:
            row_str_stretched = '\t'.join(str(item) for item in row_stretched)
            table_file_stretched.write(f"{row_str_stretched}\n")

    print("Fit results tables generated successfully.")

    
