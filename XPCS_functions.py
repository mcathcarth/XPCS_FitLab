import tkinter as tk
from qtpy.QtWidgets import QApplication, QFileDialog
import sys
import h5py
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import warnings
from scipy.stats import pearsonr
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

    def set_sirius_new():
        nonlocal synchrotron
        synchrotron = 'Sirius new'
        root.destroy()
        
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

    sirius_new_button = tk.Button(root, text="Sirius new", command=set_sirius_new)
    sirius_new_button.pack(pady=5)
    
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
        'q_values': 0,
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
    elif selected_synchrotron == 'Sirius new':
        var = hdf5_file.split('_')
        n = var[-1].split('.')[0]
        base_name = '_'.join(var[:-3])+'_'+n
    elif selected_synchrotron == 'APS':
        base_name = hdf5_file.replace('.hdf5', '')

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

    # Handle exceptions that may occur during data processing
    except (OSError, KeyError) as e:
        return None, e
    
    return multi_tau, dataset_keys

# Function to process data from sirius
def process_sirius_data(file_path):
    """
    Process data from an HDF5 file of 'Sirius' and generate a DataFrame with columns 't' and 'g2(q=X)'.

    Args:
        file_path (str): The path to the HDF5 file.

    Returns:
        pd.DataFrame: A DataFrame with columns 't' and 'g2(q=X)' for different 'q' values.
    """
    try:
        # Open the HDF5 file
        with h5py.File(file_path, 'r') as hdf:
            # Get the dataset keys and multi-tau data
            multi_tau, dataset_keys = process_multi_tau_sirius(hdf)

            # Handle exceptions that may occur during data processing
            if multi_tau is None:
                error_message = f"### Error ###:\nThe file does not belong to 'Sirius' or there is an issue with the data.\n{dataset_keys}"
                print(error_message)
                return None
            
            # Initialize empty DataFrame and lists for relax_rate vs q^2 values
            combined_df = pd.DataFrame()
            
            # Iterate over the dataset keys
            for i, dataset_key in enumerate(dataset_keys):
                
                # Remove leading and trailing whitespace
                dataset_name = dataset_key.strip()
                
                # Extract data from the dataset within Multi-tau
                dataset = multi_tau.get(dataset_key)
                if dataset is None:
                    continue
                
                column = dataset[:]
                
                # Check if the data has a single column
                if column.ndim != 1:
                    continue
                
                # Get the column names from the HDF5 file
                column_names = list(column.dtype.names)
                
                # Get the values of the columns
                t = column['delay time (s)']
                g2 = column['g2']
                
                # Check for invalid values in the data using a boolean mask
                invalid_mask = np.isnan(t) | np.isnan(g2) | np.isinf(t) | np.isinf(g2)
                if invalid_mask.any():
                    continue
                
                # Remove the first data point
                t = t[1:]
                g2 = g2[1:]
                
                # Get the 'q' value from the dataset name
                match = re.search(r"q = (\S+)", dataset_name)
                if match:
                    q_value = float(match.group(1))
                else:
                    q_value = np.nan
                    
                # Add 'g2' data as a column in the DataFrame with the name 'g2(q=X)'
                combined_df[f"g2(q={q_value})"] = g2
                
            # Insert the 't' column as the first column (column 0)
            combined_df.insert(0, '# t', t)
            
        return combined_df

    # Handle exceptions that may occur during data processing
    except (OSError, KeyError) as e:
        error_message = f"### Error ###:\nThe file does not belong to 'Sirius' or there is an issue with the data.\n{e}"
        print(error_message)
        return None
    
# Function to process data from sirius_new
def process_sirius_new_data(file_path):
    """
    Process data from an HDF5 file of 'Sirius new' and generate a DataFrame with columns 't' and 'g2(q=X)' for each 'q'.

    Args:
        file_path (str): The path to the HDF5 file.

    Returns:
        pd.DataFrame: A DataFrame with columns 't' and 'g2(q=X)' for different 'q' values.
    """
    try:
        with h5py.File(file_path, 'r') as hdf:
            # Navigate to the 'Fitting results' directory
            fitting_results = hdf['Fitting results']

            # Get the 'q' values from the dataset 'q values (angstroms)'
            q_values = fitting_results['q values (angstroms)'][:]

            # Navigate to the 'One-tome correlation function' directory
            one_time_corr_function = hdf['One-time correlation function']

            # Get the 't' data from the first column
            t_data = one_time_corr_function['1-TCF'][:, 0]

            # Initialize an empty DataFrame with 't' as the first column
            combined_df = pd.DataFrame({'# t': t_data})

            # Add 'g2' data for each 'q' to the DataFrame
            for i in range(len(q_values)):
                q_value = q_values[i]
                g2_data = one_time_corr_function['1-TCF'][:, i + 1]
                combined_df[f'g2(q={q_value})'] = g2_data

        return combined_df
    
    # Handle exceptions that may occur during data processing
    except (OSError, KeyError) as e:
        error_message = f"### Error ###:\nThe file does not belong to 'Sirius new' or there is an issue with the data.\n{e}"
        print(error_message)
        return None
        
# Function to process data from APS
def process_aps_data(file_path):
    """
    Process data from an HDF5 file of 'APS' and generate a DataFrame with columns 't' and 'g2(q=X)'.

    Args:
        file_path (str): The path to the HDF5 file.

    Returns:
        pd.DataFrame: A DataFrame with columns 't' and 'g2(q=X)' for different 'q' values.
    """
    return pd.DataFrame()

# Function to initialize R tracking for parameter averages
def initialize_data_for_parameter_averages():
    """
    Initializes data structures for storing data to calculate parameter averages.
    
    Returns:
        dict: A dictionary containing counters and sums for R values and parameters.
    """
    av_params = {
        'single': {
            'R count': 0,
            'Diff_coef sum': 0,
            'Diff_coef values': []
        },
        'stretched': {
            'R count': 0,
            'Diff_coef sum': 0,
            'Diff_coef values': [],
            'Gamma sum': 0,
            'Gamma values': []
        },
        'cumulants': {
            'R count': 0,
            'Diff_coef sum': 0,
            'Diff_coef values': [],
            'PDI sum': 0,
            'PDI values': []
        }
    }
    return av_params

# Function to initialize a plot
def initialize_plot(q_len):
    """
    Initialize a matplotlib plot with specified LaTeX font and other settings.

    Args:
        q_len (int): The length of the 'i_values' list for determining the color map size.

    Returns:
        tuple: A tuple containing figure, axes, color map, and a dictionary of lists for organizing data.
               The dictionary keys include:
               - 'exp_lines_single', 'exp_labels_single', 
                 'fit_lines_single', 'fit_labels_single'
               - 'exp_lines_stretched', 'exp_labels_stretched', 
                 'fit_lines_stretched', 'fit_labels_stretched'
               - 'exp_lines_cumulants', 'exp_labels_cumulants', 
                 'fit_lines_cumulants', 'fit_labels_cumulants'
    """
    
    # Configure LaTeX font with Cambria Math
    #plt.rcParams["text.usetex"] = True
    #plt.rcParams["text.latex.preamble"] = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{fontspec} \setmainfont{Cambria Math}'
    
    # Create a new figure for each HDF5 file with a 2x2 subplot grid
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 16))

    # Access individual subplots using the axes array
    ax_single = axes[0, 0]
    ax_stretched = axes[0, 1]
    ax_cumulants = axes[1, 0]  # New axis for the Cumulants model
    ax_linear = axes[1, 1]
    
    # Initialize a color map for different q values
    #cmap = custom_cmap()
    #cmap = plt.get_cmap('tab10')
    cmap = plt.cm.get_cmap('gist_rainbow', q_len)
    #cmap = plt.cm.get_cmap('turbo', q_len)

    # Initialize a dictionary to store lines and labels for legend items
    lines_labels_dict = {
        "exp_lines_single": [],    # Lines for experimental data for Single Exponential
        "exp_labels_single": [],   # Labels for experimental data for Single Exponential
        "fit_lines_single": [],    # Lines for fitted curves for Single Exponential
        "fit_labels_single": [],   # Labels for fitted curves for Single Exponential
        "exp_lines_stretched": [],  # Lines for experimental data for Stretched Exponential
        "exp_labels_stretched": [], # Labels for experimental data for Stretched Exponential
        "fit_lines_stretched": [],  # Lines for fitted curves for Stretched Exponential
        "fit_labels_stretched": [],  # Labels for fitted curves for Stretched Exponential
        "exp_lines_cumulants": [],  # Lines for experimental data for Cumulants
        "exp_labels_cumulants": [],  # Labels for experimental data for Cumulants
        "fit_lines_cumulants": [],  # Lines for fitted curves for Cumulants
        "fit_labels_cumulants": []  # Labels for fitted curves for Cumulants
    }
    
    return fig, ax_single, ax_stretched, ax_cumulants, ax_linear, cmap, lines_labels_dict

# Function to initialize empty data structures for derived parameters
def initialize_data_for_derived_params():
    """
    Initializes data structures for derived parameters, including Relaxation time (s) 
    and Diffusion coefficient (u^2/s).
   
    Returns:
        dict: A dictionary with keys for derived parameters, 
              each containing an empty list.
    """
    # Initialize empty lists for relax_rate vs q^2 values
    derived_params = {
        "q_square": [],   # Empty list for q_square model values
        "relax_rate": [],  # Empty list for relax_rate model values
    }
        
    return derived_params

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

# Function to fit a given model to the data and return parameters, fitted curve, and R2 score
def fit_model_with_constraints(t, g2, model_func, initial_params):
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

            # Restriction for parameter A based on the model
            if model_func.__name__ == 'single_exponential':
                bounds = ([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf])  # A>0
            elif model_func.__name__ == 'stretched_exponential':
                bounds = ([0, -np.inf, -np.inf, 0], [np.inf, np.inf, np.inf, np.inf])  # A>0 gamma>0 
            elif model_func.__name__ == 'cumulants_model':
                bounds = ([0, -np.inf, -np.inf, 0], [np.inf, np.inf, np.inf, np.inf])  # A>0 C2>0

            fit_params, _ = curve_fit(model_func, t, g2, p0=initial_params, bounds=bounds)
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

# Cumulants Model function
def cumulants_model(t, A, B, C1, C2):
    """
    Cumulants Model function for g^2(tau).

    Parameters:
        t (array-like): Los valores de tiempo.
        A (float): Parámetro A.
        B (float): Parámetro B.
        C1 (float): Parámetro C1.
        C2 (float): Parámetro C2.

    Returns:
        array-like: The fitted curve for g^2(tau).
    """
    return A + B * np.exp(-2 * C1 * t) * (1 + (1 / 2) * C2 * t**2)**2

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
def write_dat_file(output_path, q_value, t, g2, A_single, B_single, C_single, A_stretched, B_stretched, C_stretched, gamma,
                   r2_single, r2_stretched, relax_time_single, relax_time_stretched, diffusion_coef_single, diffusion_coef_stretched,
                   A_cumulants, B_cumulants, C1_cumulants, C2_cumulants, r2_cumulants, relax_time_cumulants, diffusion_coef_cumulants):
    """
    Write data and comments to a .dat file.

    Parameters:
        output_path (str): The path to the output .dat file.
        q_value (float): The q-value associated with the data.
        t (array-like): Time values.
        g2 (array-like): Experimental data.
        A_single, B_single, C_single, A_stretched, B_stretched, C_stretched, gamma, A_cumulants, B_cumulants, C1_cumulants, C2_cumulants (float): Fit parameters for Single Exponential, Stretched Exponential and Cumulants Model.
        r2_single, r2_stretched, r2_cumulant (float): R2 scores for Single Exponential, Stretched Exponential and Cumulants Model.
        relax_time_single, relax_time_stretched, relax_time_cumulants (float): Relaxation times for Single Exponential, Stretched Exponential and Cumulants Model.
        diffusion_coef_single, diffusion_coef_stretched, diffusion_coef_cumulants (float): Diffusion coefficients for Single Exponential, Stretched Exponential and Cumulants Model.
    """
    with open(output_path, 'w') as dat_file:
        dat_file.write(f"q = {q_value} A^-1\n")
        dat_file.write("### Fit model: Single Exponential ###\n")
        dat_file.write(f"#Fit parameters: baseline = {A_single}, beta = {B_single}, relax rate (1/s) = {C_single}\n")
        dat_file.write(f"#R2: {r2_single}\n")
        dat_file.write(f"#Derived parameters: Relax. time (s) = {relax_time_single}, Diffusion coef (u2/s) = {diffusion_coef_single}\n")
        dat_file.write("### Fit model: Stretched Exponential ###\n")
        dat_file.write(f"#Fit parameters: baseline = {A_stretched}, beta = {B_stretched}, relax rate (1/s) = {C_stretched}, gamma = {gamma}\n")
        dat_file.write(f"#R2: {r2_stretched}\n")
        dat_file.write(f"#Derived parameters: Relax. time (s) = {relax_time_stretched}, Diffusion coef (u2/s) = {diffusion_coef_stretched}\n")
        dat_file.write("### Fit model: Cumulants Model ###\n")
        dat_file.write(f"#Fit parameters: baseline = {A_cumulants}, beta = {B_cumulants}, relax rate (1/s) = {C1_cumulants}, PDI = {C2_cumulants/C1_cumulants**2}\n")
        dat_file.write(f"#R2: {r2_cumulants}\n")
        dat_file.write(f"#Derived parameters: Relax. time (s) = {relax_time_cumulants}, Diffusion coef (u2/s) = {diffusion_coef_cumulants}\n")
        dat_file.write("#delay time (s)\tg2\tstd\tFitted g2 Single\tFitted g2 Stretched\tFitted g2 Cumulants\n")

        fitted_curve_single = A_single + B_single * np.exp(-2 * C_single * t)
        fitted_curve_stretched = A_stretched + B_stretched * np.exp(-2 * C_stretched * t) ** gamma
        fitted_curve_cumulants = A_cumulants + B_cumulants * np.exp(-2 * C1_cumulants * t) * (1 + (1 / 2) * C2_cumulants * t**2)**2

        for i in range(len(t)):
            row_str = f"{t[i]}\t{g2[i]}\t{np.std(g2)}\t{fitted_curve_single[i]}\t{fitted_curve_stretched[i]}\t{fitted_curve_cumulants[i]}\n"
            dat_file.write(row_str)

# Function to plot experimental data and fitted curves
def plot_data_and_curves(ax, t, g2, baseline, Beta, fitted_curve, q_value, r2_value, lines_labels_dict, color, linestyle, model_type):
    """
    Add experimental data and fitted curves to a plot.

    Args:
        ax (matplotlib.axes.Axes): The Axes object to which data will be added.
        t (numpy.ndarray): Array of delay times.
        g2 (numpy.ndarray): Array of experimental g2 values.
        baseline (float): Value of the fitting parameter baseline (e.g., A_single)
        Beta (float): Value of the fitting parameter Beta (e.g., B_single B_stretched or B_cumulants).
        fitted_curve (numpy.ndarray): Fitted curve.
        q_value (float): Value of q.
        r2_value (float): R-squared value.
        lines_labels_dict (dict): Dictionary containing lists for organizing data.
        color (str): The color for the plot elements.
        linestyle (str, optional): The linestyle for the fitted curve. Defaults to '-'.
        model_type (str): The type of model, either "Single", "Stretched" or "Cumulants".

    Returns:
        None
    """
    # Plot experimental data points
    line_exp = ax.semilogx(t, (g2 - baseline)/Beta, 'o', color=color)[0]

    # Plot fitted curve
    line_fit = ax.semilogx(t, (fitted_curve - baseline)/Beta, color=color, linestyle=linestyle)[0]

    # Add labels to the legend lists based on the model type
    lines_labels_dict[f"exp_lines_{model_type.lower()}"].append(line_exp)
    lines_labels_dict[f"exp_labels_{model_type.lower()}"].append(f"q = {q_value:.6f}")
    lines_labels_dict[f"fit_lines_{model_type.lower()}"].append(line_fit)
    lines_labels_dict[f"fit_labels_{model_type.lower()}"].append(f"R2: {r2_value:.3f}")

# Function to update parameter tracking
def update_parameter_tracking_r(parameter_data, model, r_value, parameter_value1, parameter_value2=None):
    """
    Updates the parameter tracking counters and sums for a specific model.

    Args:
        parameter_data (dict): A dictionary containing counters and sums for parameter values.
        model (str): The model name ('single', 'stretched', or 'cumulants').
        r_value (float): The R value for the current adjustment.
        parameter_value1 (float): The first parameter value for the current adjustment.
        parameter_value2 (float, optional): The second parameter value for the current adjustment (only for models requiring two parameters).
    """
    parameter_data[model]['R count'] += 1
    parameter_data[model]['Diff_coef sum'] += parameter_value1
        
    if model == 'stretched' and parameter_value2 is not None:
        parameter_data[model]['Gamma sum'] += parameter_value2
    elif model == 'cumulants' and parameter_value2 is not None:
        parameter_data[model]['PDI sum'] += parameter_value2

    # Guardar el valor individual en una lista para el cálculo del desvío estándar
    parameter_data[model]['Diff_coef values'].append(parameter_value1)
    
    if model == 'stretched' and parameter_value2 is not None:
        parameter_data[model]['Gamma values'].append(parameter_value2)
    elif model == 'cumulants' and parameter_value2 is not None:
        parameter_data[model]['PDI values'].append(parameter_value2)

# Define the function that will perform the linear fit and return the parameters
def fit_linear_model(q_squared_values, C_values):
    """
    Fit a linear model C = D * q^2 to given data.
    
    Parameters:
    q_squared_values (array-like): Array of q^2 values.
    C_values (array-like): Array of corresponding relaxation rates.
    
    Returns:
    D (float): Diffusion coefficient obtained from the linear fit.
    pearson_r (float): Pearson correlation coefficient of the fit.
    """
    # Convert q_squared_values to float values
    q_squared_values = np.array(q_squared_values, dtype=np.float64)
    
    # Define the linear function to fit
    def linear_function(q_squared, D):
        return D * q_squared + 0  # Force y-intercept to be zero

    try:
       # Perform the linear fit
       fit_params, cov_matrix = curve_fit(linear_function, q_squared_values, C_values)
       
       # Get the parameter D (slope of the linear fit)
       D = fit_params[0]
       
       # Calculate the Pearson correlation coefficient
       pearson_r, _ = pearsonr(C_values, linear_function(q_squared_values, D))
       
       return D, pearson_r
   
    except (RuntimeError, ValueError):
        # Handle fitting errors, e.g., NaN or infinite values
        return np.nan, np.nan
    
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
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    #ax.set_ylabel(ylabel, fontsize=14)
    ax.set_ylabel(ylabel.replace("Beta", "β"), fontsize=14)

    # Create the legend for experimental data
    legend_exp = ax.legend(exp_lines, exp_labels, loc='upper right', bbox_to_anchor=(0.86, 1), borderaxespad=0)
    ax.add_artist(legend_exp)

    # Create the legend for fitted curves
    legend_fit = ax.legend(fit_lines, fit_labels, loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0)
    ax.add_artist(legend_fit)

# Function to calculate average parameter values
def calculate_average_parameter_values(param_dict, model):
    """
    Calculates the average parameter values and standard deviation for different models based on parameter tracking.

    Args:
        param_dict (dict): A dictionary containing counters and sums for parameter values.
        model (str): The key for which to calculate average values.

    Returns:
        dict: A dictionary containing the calculated average values and standard deviation.
    """
    average_parameters = {}

    if model in param_dict:
        parameters = param_dict[model]

        # Calculate the average Diff_coef value
        average_parameters['Diff_coef av'] = parameters['Diff_coef sum'] / parameters['R count']
        
        # Calculate the standard deviation for Diff_coef
        diff_coef_values = parameters['Diff_coef values']
        diff_coef_std = np.std(diff_coef_values)
        average_parameters['Diff_coef std'] = diff_coef_std

        if model == 'stretched':
            # Calculate the average Gamma value
            average_parameters['Gamma av'] = parameters['Gamma sum'] / parameters['R count']

            # Calculate the standard deviation for Gamma
            gamma_values = parameters['Gamma values']
            gamma_std = np.std(gamma_values)
            average_parameters['Gamma std'] = gamma_std

        elif model == 'cumulants':
            # Calculate the average PDI value
            average_parameters['PDI av'] = parameters['PDI sum'] / parameters['R count']

            # Calculate the standard deviation for PDI
            pdi_values = parameters['PDI values']
            pdi_std = np.std(pdi_values)
            average_parameters['PDI std'] = pdi_std

    return average_parameters
            
# Function to print error and success counters
def print_summary(counters):
    """
    Print a summary of processing results.

    Parameters:
        counters (dict): A dictionary containing counters and lists for error and success tracking.
    """
    print(f"Total HDF5 files: {counters['hdf5_files']}")
    print(f"Total x files: {counters['q_values']}")
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
def generate_fit_results_tables(directory, table_data_single, table_data_stretched, table_data_cumulants):
    """
    Generate .dat files with fit results tables for Single Exponential, Stretched Exponential, and Cumulants models.

    Parameters:
        directory (str): Directory where the .dat files will be saved.
        table_data_single (list): List of lists containing data for Single Exponential table.
        table_data_stretched (list): List of lists containing data for Stretched Exponential table.
        table_data_cumulants (list): List of lists containing data for Cumulants table.
    """
    table_output_path_single = os.path.join(directory, "fit_results_single.dat")
    table_output_path_stretched = os.path.join(directory, "fit_results_stretched.dat")
    table_output_path_cumulants = os.path.join(directory, "fit_results_cumulants.dat")

    with open(table_output_path_single, 'w') as table_file_single, open(table_output_path_stretched, 'w') as table_file_stretched, open(table_output_path_cumulants, 'w') as table_file_cumulants:
        # Write header for Single Exponential
        header_single = "{:^26} {:^13} {:^23} {:^24} {:^24} {:^26} {:^25} {:^24}\n".format(
            "Filename", "q (A^-1)", "Baseline", "beta", "Relax. rate (s-1)", "Relax. time (s)", "Diff. coefficient (u2/s)", "R2"
        )
        table_file_single.write(header_single)

        # Write header for Stretched Exponential
        header_stretched = "{:^26} {:^13} {:^23} {:^24} {:^24} {:^24} {:^26} {:^25} {:^24}\n".format(
            "Filename", "q (A^-1)", "Baseline", "beta", "Relax. rate (s-1)", "Gamma", "Relax. time (s)", "Diff. coefficient (u2/s)", "R2"
        )
        table_file_stretched.write(header_stretched)

        # Write header for Cumulants
        header_cumulants = "{:^26} {:^13} {:^23} {:^24} {:^24} {:^24} {:^26} {:^25} {:^24}\n".format(
            "Filename", "q (A^-1)", "Baseline", "beta", "Relax. rate (s-1)", "PDI", "Relax. time (s)", "Diff. coefficient (u2/s)", "R2"
        )
        table_file_cumulants.write(header_cumulants)

        # Write table data for Single Exponential
        for row_single in table_data_single:
            row_str_single = "{:<26} {:<13} {:<23} {:<24} {:<24} {:<26} {:<25} {:<24}\n".format(*row_single)
            table_file_single.write(row_str_single)

        # Write table data for Stretched Exponential
        for row_stretched in table_data_stretched:
            row_str_stretched = "{:<26} {:<13} {:<23} {:<24} {:<24} {:<24} {:<26} {:<25} {:<24}\n".format(*row_stretched)
            table_file_stretched.write(row_str_stretched)

        # Write table data for Cumulants
        for row_cumulants in table_data_cumulants:
            row_str_cumulants = "{:<26} {:<13} {:<23} {:<24} {:<24} {:<24} {:<26} {:<25} {:<24}\n".format(*row_cumulants)
            table_file_cumulants.write(row_str_cumulants)

    print("Fit results tables generated successfully.")

