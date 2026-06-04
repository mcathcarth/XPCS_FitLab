# XPCS_functions.py - Module containing essential functions for XPCS data analysis.

# Author: Marilina Cathcarth [mcathcarth@gmail.com]
# Version: 6.8
# Date: June, 2026

import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import os
import h5py
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from functools import partial
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import warnings
from scipy.stats import pearsonr
from functools import partial
#from scipy.optimize import minimize

#-----------------------------------------------------------------------#
#--------------------- Directory and File Handling ---------------------#
#-----------------------------------------------------------------------#

# Function to select synchrotron
def select_synchrotron():
    """
    Displays a GUI window to select a synchrotron and returns the selected synchrotron and data type.
    
    Returns:
        tuple: A tuple containing the selected synchrotron and data type.
    """
    # Initialize variables
    synchrotron = None
    data_type = None

    # Define the nested functions to set the synchrotron:
    ### SIRIUS ###
    def set_sirius():
        nonlocal synchrotron         # Access the outer variable
        synchrotron = 'Sirius'       # Set the synchrotron to 'Sirius'
        open_sirius_versions()       # Open window to select Sirius versions

    def open_sirius_versions():
        # Create a new window to select Sirius versions
        sirius_window = tk.Toplevel(root)
        sirius_window.title("Select Sirius Version")

        # Set the window size and position it in the center of the screen
        sirius_window_width = 300
        sirius_window_height = 210
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        sirius_x_position = (screen_width - window_width) // 2
        sirius_y_position = (screen_height - window_height) // 2

        sirius_window.geometry(f"{sirius_window_width}x{sirius_window_height}+{sirius_x_position}+{sirius_y_position}")
        #sirius_window.geometry(root.geometry())  # Use the same geometry as the root window

        # Create the label widget to display the instruction
        label = tk.Label(sirius_window, text="Select the Sirius version:")
        label.pack(pady=10)                 # Add some padding

        # Create the buttons to select Sirius versions
        sirius1_button = tk.Button(sirius_window, text="version 1 (May, 2023)", command=set_sirius_1)
        sirius1_button.pack(pady=5)

        sirius2_button = tk.Button(sirius_window, text="version 2 (October, 2023)", command=set_sirius_2)
        sirius2_button.pack(pady=5)

        sirius3_button = tk.Button(sirius_window, text="version 3 (January, 2024)", command=set_sirius_3)
        sirius3_button.pack(pady=5)

        sirius3m_button = tk.Button(sirius_window, text="version 3 (Average)", command=set_sirius_3m)
        sirius3m_button.pack(pady=5)

        # Bind the closing of the window to the on_closing function
        sirius_window.protocol("WM_DELETE_WINDOW", on_closing)

    def select_data_type(type):
        nonlocal data_type
        data_type = type
        root.destroy()

    def open_data_type_selection():
        # Create a new window to select data type
        data_type_window = tk.Toplevel(root)
        data_type_window.title("Select Data Type")
        # Set the window size and position it in the center of the screen
        data_type_window_width = 200
        data_type_window_height = 150
        data_type_screen_width = root.winfo_screenwidth()
        data_type_screen_height = root.winfo_screenheight()
        data_type_x_position = (data_type_screen_width - data_type_window_width) // 2
        data_type_y_position = (data_type_screen_height - data_type_window_height) // 2

        data_type_window.geometry(f"{data_type_window_width}x{data_type_window_height}+{data_type_x_position}+{data_type_y_position}")

        # Create a label widget to display the instruction
        label = tk.Label(data_type_window, text="Select data type:")
        label.pack(pady=10)                 # Add some padding

        # Create the buttons to select data type
        originals_button = tk.Button(data_type_window, text="Original Data", command=lambda: select_data_type("Originals"))
        originals_button.pack(pady=5)

        rebineds_button = tk.Button(data_type_window, text="Rebinned Data", command=lambda: select_data_type("Rebinneds"))
        rebineds_button.pack(pady=5)

    def set_sirius_1():
        nonlocal synchrotron
        synchrotron = 'Sirius 1'
        root.destroy()

    def set_sirius_2():
        nonlocal synchrotron, data_type
        synchrotron = 'Sirius 2'
        root.destroy()

    def set_sirius_3():
        nonlocal synchrotron, data_type
        synchrotron = 'Sirius 3'
        open_data_type_selection()
        #root.destroy()

    def set_sirius_3m():
        nonlocal synchrotron, data_type
        synchrotron = 'Sirius 3 average'
        open_data_type_selection()

    ### APS ###
    def set_APS():
        nonlocal synchrotron
        synchrotron = 'APS'
        root.destroy()

    ### ESRF ###
    def set_ESRF():
        nonlocal synchrotron
        synchrotron = 'ESRF'
        root.destroy()

    # Set synchrotron to None when the window is closed
    def on_closing():
        nonlocal synchrotron
        synchrotron = None
        root.destroy()

    # Create the main GUI window
    root = tk.Tk()
    root.title("Select Synchrotron")    # Set the title of the window

    # Set the window size and position it in the center of the screen
    window_width = 300
    window_height = 180
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x_position = (screen_width - window_width) // 2
    y_position = (screen_height - window_height) // 2

    root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

    # Create a label widget to display the instruction
    label = tk.Label(root, text="Select the synchrotron:")
    label.pack(pady=10)                 # Add some padding

    # Create the buttons to select the synchrotron 
    sirius_button = tk.Button(root, text="Sirius", command=set_sirius)
    sirius_button.pack(pady=5)

    APS_button = tk.Button(root, text="APS", command=set_APS)
    APS_button.pack(pady=5)

    ESRF_button = tk.Button(root, text="ESRF", command=set_ESRF)
    ESRF_button.pack(pady=5)

    # Bind the closing of the window to the on_closing function
    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Start the main GUI event loop
    root.mainloop()

    return synchrotron, data_type

# Function to select a directory or files
def select_directory_or_files():
    """
    Display a Tkinter window to select either a directory or individual HDF5 files.

    Returns:
        tuple:
            - directory_path (str): Path to the selected directory.
            - selected_files (list): List of selected HDF5 filenames.
    """
    directory_path = ""
    selected_files = []

    def center_window(window, width=300, height=130):
        """Center a Tkinter window on the screen."""
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x_position = (screen_width - width) // 2
        y_position = (screen_height - height) // 2
        window.geometry(f"{width}x{height}+{x_position}+{y_position}")

    def close_window():
        """Close the selection window safely."""
        try:
            root.destroy()
        except tk.TclError:
            pass

    def select_directory():
        """Select a directory and collect all HDF5 files inside it."""
        nonlocal directory_path, selected_files

        selected_path = filedialog.askdirectory(
            parent=root,
            title="Select Directory"
        )

        if selected_path:
            directory_path = selected_path
            selected_files = [
                os.path.basename(file)
                for file in os.listdir(directory_path)
                if file.lower().endswith((".hdf5", ".h5"))
            ]
        else:
            directory_path = ""
            selected_files = []

        close_window()

    def select_files():
        """Select individual HDF5 files."""
        nonlocal directory_path, selected_files

        selected_file_paths = filedialog.askopenfilenames(
            parent=root,
            title="Select Files",
            filetypes=[
                ("HDF5 files", "*.hdf5 *.h5"),
                ("All files", "*.*")
            ]
        )

        if selected_file_paths:
            directory_path = os.path.dirname(selected_file_paths[0])
            selected_files = [
                os.path.basename(file_path)
                for file_path in selected_file_paths
            ]
        else:
            directory_path = ""
            selected_files = []

        close_window()

    def cancel_selection():
        """Cancel file selection."""
        nonlocal directory_path, selected_files
        directory_path = ""
        selected_files = []
        close_window()

    # Create the selection window
    root = tk.Tk()
    root.title("Select Directory or Files")
    root.resizable(False, False)
    center_window(root)

    # Instruction label
    label = tk.Label(root, text="Select the input source:")
    label.pack(pady=10)

    # Buttons
    directory_button = tk.Button(
        root,
        text="Select Directory",
        width=20,
        command=select_directory
    )
    directory_button.pack(pady=4)

    files_button = tk.Button(
        root,
        text="Select Files",
        width=20,
        command=select_files
    )
    files_button.pack(pady=4)

    root.protocol("WM_DELETE_WINDOW", cancel_selection)

    root.mainloop()

    return directory_path, selected_files

# Function to prompt the user for defining the range
def ask_user_for_t_range():
    """
    Ask the user if they want to define the range of 't'.

    Returns:
        bool: True if the user chooses 'Yes', False otherwise.
    """
    root = tk.Tk()
    root.withdraw()

    user_response = messagebox.askyesno(
        "Define Range",
        "Do you want to define the range of 't'?",
        parent=root
    )

    root.destroy()

    return user_response

# Function to generate the base name
def generate_base_name(selected_synchrotron, hdf5_file):
    """
    Generates a base name for data processing based on the selected synchrotron.

    Parameters:
        selected_synchrotron (str): The selected synchrotron name.
        hdf5_file (str): The HDF5 file name.

    Returns:
        str: The generated base name.
    """
    if selected_synchrotron == 'Sirius 1':
        base_name = hdf5_file.replace('_saxs_0000_RESULTS.hdf5', '')
    elif selected_synchrotron == 'Sirius 2':
        var = hdf5_file.split('_')
        n = var[-1].split('.')[0]
        base_name = '_'.join(var[:-3])+'_'+n
    elif selected_synchrotron == 'Sirius 3':
        var = hdf5_file.split('_')
        n = var[-1].split('.')[0]
        base_name = '_'.join(var[:-3])+'_'+n
    elif selected_synchrotron == 'Sirius 3 average':
        base_name = hdf5_file.replace('.hdf5', '')
    elif selected_synchrotron == 'APS':
        base_name = hdf5_file.replace('.hdf5', '')
    elif selected_synchrotron == 'ESRF':
        base_name = os.path.splitext(hdf5_file)[0]

    return base_name

# Function to process 'Multi-tau' data for Sirius
def process_multi_tau_sirius1(hdf):
    """
    Process the 'Multi-tau' data for Sirius version 1.

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

# Function to process data from sirius_v1
def process_sirius1_data(file_path):
    """
    Process data from an HDF5 file of 'Sirius version 1' and generate a DataFrame with columns 't' and 'g2(q=X)'.

    Args:
        file_path (str): The path to the HDF5 file.

    Returns:
        pd.DataFrame: A DataFrame with columns 't' and 'g2(q=X)' for different 'q' values.
    """
    try:
        # Open the HDF5 file
        with h5py.File(file_path, 'r') as hdf:
            # Get the dataset keys and multi-tau data
            multi_tau, dataset_keys = process_multi_tau_sirius1(hdf)

            # Handle exceptions that may occur during data processing
            if multi_tau is None:
                error_message = f"### Error ###:\nThe file does not belong to 'Sirius version 1' or there is an issue with the data.\n{dataset_keys}"
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
    
# Function to process data from sirius_v2
def process_sirius2_data(file_path):
    """
    Process data from an HDF5 file of 'Sirius version 2' and generate DataFrames.

    Args:
        file_path (str): The path to the HDF5 file.
        
    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: A DataFrame with columns 't' and 'g2(q=X)' for each 'q' values.
            - dict: A dictionary containing DataFrames with columns 't1' and 't2' for each 'q' value.
    """
    try:
        with h5py.File(file_path, 'r') as hdf:
            ### q values ###
            # Navigate to the 'Fitting results' directory
            fitting_results = hdf['Fitting results']

            # Get the 'q' values from the dataset 'q values (angstroms)'
            q_values = fitting_results['q values (angstroms)'][:]

            ### t data ###
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

            ### Two-time corralation ###
            # Navigate to the 'Two-time correlation' directory
            two_time_corr_function = hdf['Two-time correlation function']['2-TCF']

            # Initialize dictionary to store t1 vs t2 DataFrames for each q
            t1t2_dataframes = {}

            # Iterate over the datasets within the group and the corresponding q values
            for q, t1t2_dataset in zip(q_values, two_time_corr_function):
                # Convert the dataset into a pandas DataFrame and transpose it
                t1t2_df = pd.DataFrame(t1t2_dataset[:])
                # Add the transposed DataFrame to the dictionary with q as the key
                t1t2_dataframes[q] = t1t2_df

        return combined_df, t1t2_dataframes
    
    # Handle exceptions that may occur during data processing
    except (OSError, KeyError) as e:
        error_message = f"### Error ###:\nThe file does not belong to 'Sirius version 2' or there is an issue with the data.\n{e}"
        print(error_message)
        return None, None
    
# Function to process data from sirius_v3 average data
def process_sirius3m_data(file_path, data_type):
    """
    Process data from an HDF5 file of 'Sirius 3' and generate DataFrames.

    Args:
        file_path (str): The path to the HDF5 file.
        data_type (str): The type of data to process. Options are 'Originals' or 'Rebineds'.

    Returns:
        pd.DataFrame: A DataFrame with columns 't' and 'g2(q=X)' for different 'q' values.
    """
    try:
        with h5py.File(file_path, 'r') as hdf:
            ### q values ###
            # Navigate to the 'Fitting results' directory
            fitting_results = hdf['Multi-tau Fitting results']

            # Get the 'q' values from the dataset 'q values (angstroms)'
            q_values = fitting_results['q values (angstroms)'][:]

            ### t data ###
            if data_type == 'Originals':
                # Navigate to the 'One-time correlation' directory
                one_time_corr_function = hdf['One-time correlation']

                # Get the 't' data from the first column
                t_data = one_time_corr_function['Average One-TCF'][:, 0]

                # Initialize an empty DataFrame with 't' as the first column
                combined_df = pd.DataFrame({'# t': t_data})

                # Add 'g2' data for each 'q' to the DataFrame
                for i in range(len(q_values)):
                    q_value = q_values[i]
                    g2_data = one_time_corr_function['Average One-TCF'][:, i + 1]
                    combined_df[f'g2(q={q_value})'] = g2_data

            elif data_type == 'Rebinneds':
                # Navigate to the 'Multi-tau correlation' directory
                multi_tau_corr_function = hdf['Multi-tau Correlation']

                # Get the 't' data from the first column
                t_data = multi_tau_corr_function['Multi-tau'][:, 0]

                # Initialize an empty DataFrame with 't' as the first column
                combined_df = pd.DataFrame({'# t': t_data})

                # Add 'g2' data for each 'q' to the DataFrame
                for i in range(len(q_values)):
                    q_value = q_values[i]
                    g2_data = multi_tau_corr_function['Average Multi-tau'][:, i + 1]
                    combined_df[f'g2(q={q_value})'] = g2_data

        return combined_df
    
    # Handle exceptions that may occur during data processing
    except (OSError, KeyError) as e:
        error_message = f"### Error ###:\nThe file does not belong to 'Sirius 3 average data' or there is an issue with the data.\n{e}"
        print(error_message)
        return None
    
# Function to process data from sirius_v3
def process_sirius3_data(file_path,data_type):
    """
    Process data from an HDF5 file of 'Sirius 3' and generate DataFrames.

    Args:
        file_path (str): The path to the HDF5 file.
        data_type (str): The type of data to process. Options are 'Originals' or 'Rebineds'.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: A DataFrame with columns 't' and 'g2(q=X)' for different 'q' values.
            - dict: A dictionary containing DataFrames with columns 't1' and 't2' for each 'q' value.
    """
    try:
        with h5py.File(file_path, 'r') as hdf:
            ### q values ###
            # Navigate to the 'Fitting results' directory
            fitting_results = hdf['Multi-tau Fitting results']

            # Get the 'q' values from the dataset 'q values (angstroms)'
            q_values = fitting_results['q values (angstroms)'][:]

            ### t data ###
            if data_type == 'Originals':
                # Navigate to the 'One-time correlation' directory
                one_time_corr_function = hdf['One-time correlation']

                # Get the 't' data from the first column
                t_data = one_time_corr_function['1-TCF'][:, 0]

                # Initialize an empty DataFrame with 't' as the first column
                combined_df = pd.DataFrame({'# t': t_data})

                # Add 'g2' data for each 'q' to the DataFrame
                for i in range(len(q_values)):
                    q_value = q_values[i]
                    g2_data = one_time_corr_function['1-TCF'][:, i + 1]
                    combined_df[f'g2(q={q_value})'] = g2_data

            elif data_type == 'Rebinneds':
                # Navigate to the 'Multi-tau correlation' directory
                multi_tau_corr_function = hdf['Multi-tau Correlation']

                # Get the 't' data from the first column
                t_data = multi_tau_corr_function['Multi-tau'][:, 0]

                # Initialize an empty DataFrame with 't' as the first column
                combined_df = pd.DataFrame({'# t': t_data})

                # Add 'g2' data for each 'q' to the DataFrame
                for i in range(len(q_values)):
                    q_value = q_values[i]
                    g2_data = multi_tau_corr_function['Multi-tau'][:, i + 1]
                    combined_df[f'g2(q={q_value})'] = g2_data

            ### Two-time corralation ###
            # Navigate to the 'Two-time correlation' directory
            two_time_corr_function = hdf['Two-time correlation']['2-TCF']

            # Initialize dictionary to store t1 vs t2 DataFrames for each q
            t1t2_dataframes = {}

            # Iterate over the datasets within the group and the corresponding q values
            for q, t1t2_dataset in zip(q_values, two_time_corr_function):
                # Convert the dataset into a pandas DataFrame and transpose it
                t1t2_df = pd.DataFrame(t1t2_dataset[:])
                # Add the transposed DataFrame to the dictionary with q as the key
                t1t2_dataframes[q] = t1t2_df

        return combined_df, t1t2_dataframes
    
    # Handle exceptions that may occur during data processing
    except (OSError, KeyError) as e:
        error_message = f"### Error ###:\nThe file does not belong to 'Sirius 3' or there is an issue with the data.\n{e}"
        print(error_message)
        return None, None

# Function to process data from sirius_v3
def process_sirius3_fit(file_path,data_type):
    """
    Process fitting results (KWW) from a 'Sirius 3' HDF5 file and generate a DataFrame.

    Args:
        file_path (str): The path to the HDF5 file.
        data_type (str): The type of data to process. Options are 'Originals' or 'Rebinneds'.

    Returns:
        pd.DataFrame or None: A DataFrame with columns 'q_values', 'Baseline', 'Beta', 'Relax time', and 'Gamma'.
                              Returns None if an error occurs.
    """
    try:
        with h5py.File(file_path, 'r') as hdf:

            if data_type == 'Originals':
                # Navigate to the 'One-time Fitting results' directory
                fitting_results = hdf['One-time Fitting results']

            elif data_type == 'Rebinneds':
                # Navigate to the 'Multi-tau Fitting results' directory
                fitting_results = hdf['Multi-tau Fitting results']

            ### q values ###
            # Get the 'q' values from the dataset 'q values (angstroms)'
            q_values = fitting_results['q values (angstroms)'][:]

            ### Parameters ###
            
            # Get Basaline
            baseline = fitting_results['Baseline'][:]

            # Get Beta
            beta = fitting_results['Beta'][:]

            # Get Relax time
            relax_time = fitting_results['Relaxation time (s)'][:]

            # Get Gamma
            gamma = fitting_results['Gamma'][:]

            # Create a DataFrame
            fit_results_df = pd.DataFrame({
                'q_values': q_values,
                'Baseline': baseline,
                'Beta': beta,
                'Relax_time': relax_time,
                'Gamma': gamma
            })

        return fit_results_df
    
    # Handle exceptions that may occur during data processing
    except (OSError, KeyError) as e:
        error_message = f"### Error ###:\nThere is an issue with the data.\n{e}"
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
    try:
        with h5py.File(file_path, 'r') as hdf:
            t_data = hdf['exchange']['tau'][:].squeeze()
            q_values = hdf['xpcs']['dqlist'][:].squeeze()
            g2_data = hdf['exchange']['g2avgFIT1'][:].squeeze()

            combined_df = pd.DataFrame({'# t': t_data})

            for i, q_value in enumerate(q_values):
                g2_data_q = g2_data[:, i]
                combined_df[f'g2(q={q_value})'] = g2_data_q

            return combined_df

    except (OSError, KeyError) as e:
        error_message = f"### Error ###:\nThe file does not belong to 'APS' or there is an issue with the data.\n{e}"
        print(error_message)
        return None

# Function to process data from ESRF
def process_esrf_data(file_path):
    """
    Process data from an HDF5 or H5 file of 'ESRF' and generate DataFrames.

    Args:
        file_path (str): The path to the HDF5 file.
        
    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: A DataFrame with columns '# t' and 'g2(q=X)' for each 'q' value.
            - dict: A dictionary containing DataFrames with rows 't1' and columns 't2' for each 'q' value. NO!
    """
    try:
        with h5py.File(file_path, 'r') as hdf:
            ### q values ###
            # Navigate to the correlations group
            esrf_full = hdf['entry_0000/correlations/full']

            # Get the 'q' values array
            q_values = esrf_full['parameters/q'][:]  # shape: (nq,)

            ### t data ###
            # Get the time axis for g2
            t_data = esrf_full['g2/lag'][:]          # shape: (ntau,)

            # Initialize an empty DataFrame with '# t' as the first column
            combined_df = pd.DataFrame({'# t': t_data})

            ### g2 data ###
            # Get the g2 matrix (nq x ntau)
            cf = esrf_full['g2/cf'][:]               # shape: (nq, ntau)

            # Add 'g2' data for each 'q' to the DataFrame
            for i in range(len(q_values)):
                q_value = q_values[i]
                g2_data = cf[i, :]
                combined_df[f'g2(q={q_value})'] = g2_data

            ### Two-time correlation ###
            # Initialize dictionary to store t1 vs t2 DataFrames for each q
            #t1t2_dataframes = {}
            # Navigate to the two-time group (if present)
            #if 'twotime' in esrf_full:
            #    twotime_group = esrf_full['twotime']

                # Read the two-time cube and its time axes
                #ttcf = twotime_group['ttcf'][:]      # shape: (nq, ntau, ntau)
                #t1   = twotime_group['lag'][:]       # shape: (ntau,)
                #t2   = twotime_group['age'][:]       # shape: (ntau,)

                # Create one DataFrame per q (rows=t1, cols=t2)
                #for i in range(len(q_values)):
                    #t1t2_df = pd.DataFrame(ttcf[i, :, :], index=t1, columns=t2)
                    #t1t2_dataframes[float(q_values[i])] = t1t2_df

        return combined_df
    
    # Handle exceptions that may occur during data processing
    except (OSError, KeyError) as e:
        error_message = f"### Error ###:\nThe file does not belong to 'ESRF' or there is an issue with the data.\n{e}"
        print(error_message)
        return None, None

# Function to process data from ESRF old
def process_esrf_data_old(file_path):
    """
    Process data from an HDF5 file of 'ESRF' and generate a DataFrame with columns 't' and 'g2(q=X)'.

    Args:
        file_path (str): The path to the HDF5 file.

    Returns:
        pd.DataFrame: A DataFrame with columns 't' and 'g2(q=X)' for different 'q' values.
    """
    try:
        with h5py.File(file_path, 'r') as hdf:
            # Navigate to the relevant directories
            entry_group = hdf['entry_0000']
            dynamix_group = entry_group['dynamix']
            correlations_group = dynamix_group['correlations']
            directions_group = correlations_group['directions']
            direction_group = directions_group['direction_0000']
            correlation_group = direction_group['correlation']

            # Get 't' data from the 'timeshift' file (assuming a single column)
            t_data = correlation_group['timeshift'][:, 0]

            # Initialize an empty DataFrame with 't' as the first column
            combined_df = pd.DataFrame({'# t': t_data})

            # Get 'q' values from the 'q_index_min_max' file
            q_index_min_max = correlations_group['q_index_min_max'][:]
            q_values = q_index_min_max[:, 1:3].mean(axis=1)

            # Add 'g2' data for each 'q' to the DataFrame
            for i, q_value in enumerate(q_values):
                g2_data = correlation_group['cf'][:, i]
                combined_df[f'g2(q={q_value})'] = g2_data

        return combined_df
    
    except (OSError, KeyError) as e:
        error_message = f"### Error ###:\nThere is an issue with the data structure in the file.\n{e}"
        print(error_message)
        return None

# Function to clean the arrays
def clean_series(t, g2, min_points=8, return_pandas=True):
    t_arr  = np.asarray(t, dtype=float)
    g2_arr = np.asarray(g2, dtype=float)
    mask = np.isfinite(t_arr) & np.isfinite(g2_arr)
    t_arr, g2_arr = t_arr[mask], g2_arr[mask]

    if t_arr.size > 0 and t_arr[0] == 0.0:
        t_arr, g2_arr = t_arr[1:], g2_arr[1:]

    if t_arr.size < min_points:
        return None, None

    if return_pandas:
        return pd.Series(t_arr), pd.Series(g2_arr)
    return t_arr, g2_arr

# Function to get de t range
def get_t_range_slider(t_values):
    """
    Display a centered GUI window for the user to select a range for 't' using sliders.

    Args:
        t_values (list): List of 't' values.

    Returns:
        tuple: Lower and upper limits of the selected range.
    """
    t_values = list(t_values)

    if len(t_values) == 0:
        return 0, 0

    lower_limit = 0
    upper_limit = len(t_values) - 1

    def center_window(window, width=400, height=190):
        """Center a Tkinter window on the screen."""
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x_position = (screen_width - width) // 2
        y_position = (screen_height - height) // 2
        window.geometry(f"{width}x{height}+{x_position}+{y_position}")

    def update_lower_label(value):
        """Update the lower-limit label."""
        index = int(float(value))
        lower_slider_label.config(text=f"{t_values[index]:.4f}")

    def update_upper_label(value):
        """Update the upper-limit label."""
        index = int(float(value))
        upper_slider_label.config(text=f"{t_values[index]:.4f}")

    def apply_range():
        """Apply the selected range and close the window."""
        nonlocal lower_limit, upper_limit

        lower_limit = int(lower_slider.get())
        upper_limit = int(upper_slider.get())

        if lower_limit > upper_limit:
            messagebox.showerror(
                "Invalid range",
                "Lower limit must be less than or equal to the upper limit.",
                parent=root
            )
            return

        root.destroy()

    def cancel_range():
        """
        Close the window without changing the default range.
        The full range is used.
        """
        nonlocal lower_limit, upper_limit
        lower_limit = 0
        upper_limit = len(t_values) - 1
        root.destroy()

    def on_arrow_key_press(event):
        """Move the sliders using arrow keys."""
        if event.keysym == "Left":
            lower_slider.set(max(0, lower_slider.get() - 1))
        elif event.keysym == "Right":
            lower_slider.set(min(len(t_values) - 1, lower_slider.get() + 1))
        elif event.keysym == "Up":
            upper_slider.set(min(len(t_values) - 1, upper_slider.get() + 1))
        elif event.keysym == "Down":
            upper_slider.set(max(0, upper_slider.get() - 1))

    # Use Tk(), not Toplevel(), to avoid creating an empty ghost window.
    root = tk.Tk()
    root.title("Select 't' Range")
    root.resizable(False, False)
    center_window(root)

    lower_slider = tk.Scale(
        root,
        from_=0,
        to=len(t_values) - 1,
        orient=tk.HORIZONTAL,
        label="◄ Lower Limit ►",
        length=300,
        showvalue=0,
        command=update_lower_label
    )
    lower_slider.set(lower_limit)
    lower_slider.pack()

    lower_slider_label = tk.Label(root, text=f"{t_values[lower_limit]:.4f}", width=12)
    lower_slider_label.pack()

    upper_slider = tk.Scale(
        root,
        from_=0,
        to=len(t_values) - 1,
        orient=tk.HORIZONTAL,
        label="▲ Upper Limit ▼",
        length=300,
        showvalue=0,
        command=update_upper_label
    )
    upper_slider.set(upper_limit)
    upper_slider.pack()

    upper_slider_label = tk.Label(root, text=f"{t_values[upper_limit]:.4f}", width=12)
    upper_slider_label.pack()

    apply_button = tk.Button(root, text="Apply Range", command=apply_range)
    apply_button.pack(pady=8)

    root.bind("<Left>", on_arrow_key_press)
    root.bind("<Right>", on_arrow_key_press)
    root.bind("<Up>", on_arrow_key_press)
    root.bind("<Down>", on_arrow_key_press)

    root.protocol("WM_DELETE_WINDOW", cancel_range)

    # Bring the window to the front.
    root.lift()
    root.attributes("-topmost", True)
    root.after(200, lambda: root.attributes("-topmost", False))

    root.mainloop()

    return lower_limit, upper_limit

# Function to get the t range with graphical preview
def get_t_range_slider_preview(dataset_df, preview_q_column=None):
    """
    Display a GUI window to select a t range with a live graphical preview.

    This function is intended for single-file processing only. The preview
    shows one g2(q,t) curve and shades the selected t range.

    No fitting is performed during slider movement, so the interface remains
    responsive.

    Args:
        dataset_df (pandas.DataFrame): DataFrame containing '# t' and g2(q=...) columns.
        preview_q_column (str, optional): Column used for the preview. If None,
                                          the first g2 column is used.

    Returns:
        tuple: Lower and upper index limits of the selected range.
    """
    if '# t' not in dataset_df.columns:
        raise ValueError("The DataFrame must contain a '# t' column.")

    q_columns = [col for col in dataset_df.columns if col != '# t']

    if len(q_columns) == 0:
        raise ValueError("The DataFrame must contain at least one g2(q=...) column.")

    if preview_q_column is None or preview_q_column not in q_columns:
        preview_q_column = q_columns[0]

    t_values = dataset_df['# t'].to_numpy(dtype=float)

    if len(t_values) == 0:
        return 0, 0

    lower_limit = 0
    upper_limit = len(t_values) - 1

    def center_window(window, width=780, height=720):
        """Center a Tkinter window on the screen."""
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x_position = (screen_width - width) // 2
        y_position = (screen_height - height) // 2
        window.geometry(f"{width}x{height}+{x_position}+{y_position}")

    def get_safe_span_limits(lo_index, hi_index):
        """
        Return x limits compatible with a logarithmic x-axis.

        If t=0 is included, the smallest positive t is used only for display.
        The returned indices are not modified.
        """
        positive_t = t_values[t_values > 0]

        if positive_t.size == 0:
            return t_values[lo_index], t_values[hi_index]

        min_positive_t = positive_t.min()

        x_left = t_values[lo_index]
        x_right = t_values[hi_index]

        if x_left <= 0:
            x_left = min_positive_t

        if x_right <= 0:
            x_right = min_positive_t

        return x_left, x_right

    def draw_preview():
        """Draw or redraw the preview plot."""
        q_col = q_column_var.get()
        g2_values = dataset_df[q_col].to_numpy(dtype=float)

        ax.clear()

        valid_mask = (
            np.isfinite(t_values)
            & np.isfinite(g2_values)
            & (t_values > 0)
        )

        ax.semilogx(
            t_values[valid_mask],
            g2_values[valid_mask],
            'o-',
            markersize=3,
            linewidth=1
        )

        lo = int(lower_slider.get())
        hi = int(upper_slider.get())

        if lo <= hi:
            x_left, x_right = get_safe_span_limits(lo, hi)
            ax.axvspan(x_left, x_right, alpha=0.25)

        ax.set_xlabel("Delay time (s)")
        ax.set_ylabel(r"$g_2(q,t)$")
        ax.set_title(f"Range preview: {q_col}")
        ax.grid(True, alpha=0.3)

        figure.tight_layout()
        canvas.draw_idle()

    def update_range_labels(*args):
        """Update labels and selected region when sliders move."""
        lo = int(lower_slider.get())
        hi = int(upper_slider.get())

        lower_slider_label.config(text=f"{t_values[lo]:.5g}")
        upper_slider_label.config(text=f"{t_values[hi]:.5g}")

        if lo > hi:
            range_label.config(
                text="Invalid range: lower limit is larger than upper limit."
            )
        else:
            range_label.config(
                text=f"Selected range: {t_values[lo]:.5g} s to {t_values[hi]:.5g} s"
            )

        draw_preview()

    def apply_range():
        """Apply the selected range and close the window."""
        nonlocal lower_limit, upper_limit

        lower_limit = int(lower_slider.get())
        upper_limit = int(upper_slider.get())

        if lower_limit > upper_limit:
            messagebox.showerror(
                "Invalid range",
                "Lower limit must be less than or equal to the upper limit.",
                parent=root
            )
            return

        root.destroy()

    def use_full_range():
        """Close the window and use the full range."""
        nonlocal lower_limit, upper_limit

        lower_limit = 0
        upper_limit = len(t_values) - 1

        root.destroy()

    root = tk.Tk()
    root.title("Select 't' Range")
    root.resizable(True, True)
    center_window(root)

    # Main plot
    figure = Figure(figsize=(7.2, 3.6))
    ax = figure.add_subplot(111)

    canvas = FigureCanvasTkAgg(figure, master=root)
    canvas.get_tk_widget().pack(
        side='top',
        fill='both',
        expand=True,
        padx=10,
        pady=10
    )

    # q-column selector
    selector_frame = ttk.Frame(root)
    selector_frame.pack(fill='x', padx=10, pady=4)

    ttk.Label(selector_frame, text="Preview q:").pack(side='left', padx=(0, 6))

    q_column_var = tk.StringVar(value=preview_q_column)

    q_selector = ttk.Combobox(
        selector_frame,
        textvariable=q_column_var,
        values=q_columns,
        state="readonly",
        width=45
    )
    q_selector.pack(side='left', fill='x', expand=True)
    q_selector.bind("<<ComboboxSelected>>", lambda event: draw_preview())

    # Lower-limit slider
    lower_slider = tk.Scale(
        root,
        from_=0,
        to=len(t_values) - 1,
        orient=tk.HORIZONTAL,
        label="Lower limit",
        length=620,
        showvalue=0,
        command=update_range_labels
    )
    lower_slider.pack(fill='x', padx=20)

    lower_slider_label = tk.Label(
        root,
        text=f"{t_values[lower_limit]:.5g}",
        width=14
    )
    lower_slider_label.pack()

    # Upper-limit slider
    upper_slider = tk.Scale(
        root,
        from_=0,
        to=len(t_values) - 1,
        orient=tk.HORIZONTAL,
        label="Upper limit",
        length=620,
        showvalue=0,
        command=update_range_labels
    )
    upper_slider.pack(fill='x', padx=20)

    upper_slider_label = tk.Label(
        root,
        text=f"{t_values[upper_limit]:.5g}",
        width=14
    )
    upper_slider_label.pack()

    range_label = tk.Label(root, text="")
    range_label.pack(pady=4)

    # Buttons
    button_frame = ttk.Frame(root)
    button_frame.pack(fill='x', padx=20, pady=10)

    full_range_button = ttk.Button(
        button_frame,
        text="Use Full Range",
        command=use_full_range
    )
    full_range_button.pack(side='left')

    apply_button = ttk.Button(
        button_frame,
        text="Apply Range",
        command=apply_range
    )
    apply_button.pack(side='right')

    root.protocol("WM_DELETE_WINDOW", use_full_range)

    # Initialize slider values after all labels/widgets exist.
    lower_slider.set(lower_limit)
    upper_slider.set(upper_limit)
    update_range_labels()

    root.lift()
    root.attributes("-topmost", True)
    root.after(200, lambda: root.attributes("-topmost", False))

    root.mainloop()

    return lower_limit, upper_limit

# Function to obtain 't' range limits
def get_t_range_limits(dataset_df, define_t_range, show_preview=False):
    """
    Get the lower and upper limits for 't'.

    Args:
        dataset_df (DataFrame): The dataset DataFrame.
        define_t_range (bool): Flag indicating if 't' range is defined.
        show_preview (bool): If True, show a graphical preview of the selected range.
                             This should be used only for single-file processing.

    Returns:
        tuple: Lower and upper limits for 't'.
    """
    if '# t' not in dataset_df.columns:
        raise ValueError("The DataFrame must contain a '# t' column.")

    if not define_t_range:
        return 0, len(dataset_df['# t']) - 1

    if show_preview:
        return get_t_range_slider_preview(dataset_df)

    t_values = dataset_df['# t'].tolist()
    return get_t_range_slider(t_values)

# Function to prompt the user for defining the q values
def ask_user_for_select_q():
    """
    Ask the user if they want to select the q values.

    Returns:
        bool: True if the user chooses 'Yes', False otherwise.
    """
    root = tk.Tk()
    root.withdraw()

    user_response = messagebox.askyesno(
        "Select q Values",
        "Do you want to select q values?",
        parent=root
    )

    root.destroy()

    return user_response

# Function to save DataFrame to a TSV file with the appropriate line terminator argument
def save_dataframe_to_tsv(dataset_df, output_dataframe):
    """
    Save a pandas DataFrame to a TSV file, handling different line terminator arguments.

    Args:
        dataset_df (pandas.DataFrame): The DataFrame to be saved.
        output_dataframe (str): The file path for the output TSV file.

    Returns:
        None

    This function tries to save the DataFrame using the 'lineterminator' argument first.
    If it fails due to a TypeError, it attempts to save the DataFrame using the 'line_terminator' argument.
    This is done to handle different pandas versions, as the line terminator argument name has changed over time.
    """
    try:
        # Try using 'lineterminator'
        dataset_df.to_csv(output_dataframe, sep='\t', index=False, lineterminator='\n', header=True)
    except TypeError:
        # If that fails, use 'line_terminator'
        dataset_df.to_csv(output_dataframe, sep='\t', index=False, line_terminator='\n', header=True)

# Function to save t1-vs-t2 data to a TSV file
def save_t1t2_data(t1t2_dfs, directory, base_name):
    """
    Save the t1-t2 DataFrames to TSV files in the 't1-vs-t2' directory.

    Args:
        t1t2_dfs (dict): A dictionary of t1-t2 DataFrames.
        directory (str): The directory path where the files will be saved.
        base_name (str): The base name for the files.

    Returns:
        None
    """
    # Create the new 't1-vs-t2' directory:
    t1t2_directory = os.path.join(directory, 't1-vs-t2')
    os.makedirs(t1t2_directory, exist_ok=True)

    for i, (q, t1t2_df) in enumerate(t1t2_dfs.items(), start=1):
        # Get the letter corresponding to the index
        letter = chr(ord('a') + i - 1)
        # Create the file name with the letter
        t1t2_file = os.path.join(t1t2_directory, f"{base_name}_t1-t2_{letter}.dat")
        # Save the DataFrame to the file with q as a comment in the first line
        with open(t1t2_file, 'w') as f:
            f.write(f"# q value: {q}\n")
            t1t2_df.to_csv(f, sep='\t', index=False, header=False)

#-----------------------------------------------------------------------#
#--------------------------- Initializations ---------------------------#
#-----------------------------------------------------------------------#

# Function to initialize error and success counters
def initialize_error_and_success_counters():
    """
    Initializes counters and lists for error, success, and fit-quality tracking.
    
    Returns:
        dict: A dictionary containing counters and lists for error, success,
              and low-quality fits.
    """
    counters = {
        'hdf5_files': 0,
        'q_values': 0,
        'invalid_data': 0,
        'success': 0,
        'failure': 0,
        'below_threshold': 0,
        'invalid_files': [],
        'failed_files': [],
        'below_threshold_files': []
    }
    
    return counters

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

# Function to initialize dictionary and list for fit parameters
def initialize_fit(fit_results_df):
    """
    Initializes the Fit parameters dictionary and the fit variations list.
    
    This function processes a pandas DataFrame containing fit results from an HDF5 file
    and creates a dictionary mapping each q-value to its corresponding fit parameters.
    Additionally, it initializes an empty list to store the percentage variations
    of fit parameters between HDF5 data and code-generated fits.
    
    Args:
        fit_results_df (pd.DataFrame): 
            A pandas DataFrame containing HDF5 fit results with the following columns:
            - 'q_values': The q-values (float).
            - 'Baseline_hdf5': Baseline parameter from HDF5 fit (float).
            - 'Beta_hdf5': Beta parameter from HDF5 fit (float).
            - 'Relax_time_hdf5': Relaxation time from HDF5 fit (float).
            - 'Gamma_hdf5': Gamma parameter from HDF5 fit (float).
    
    Returns:
        tuple: 
            - fit_parameters_dict (dict): 
                A dictionary where each key is a q-value and the value is another dictionary containing:
                - 'Baseline': Baseline parameter from HDF5.
                - 'Beta': Beta parameter from HDF5.
                - 'Relax_time': Relaxation time from HDF5.
                - 'Gamma': Gamma parameter from HDF5.
            - fit_variations (list): 
                An empty list intended to store dictionaries of percentage variations for each fit parameter per q-value.
    """
    # Initialize an empty dictionary to store the mapping of q_values to fit parameters
    fit_parameters_dict = {}
    
    # Iterate over each row in the HDF5 fit results DataFrame
    for _, row in fit_results_df.iterrows():
        # Extract the q-value
        #q = round(row['q_values'], 6)
        q = row['q_values']
        
        # Map the q-value to its corresponding fit parameters
        fit_parameters_dict[q] = {
            'Baseline': row['Baseline'],
            'Beta': row['Beta'],
            'Relax_time': row['Relax_time'],
            'Gamma': row['Gamma']
        }
    
    # Initialize an empty list to store the fit parameter variations
    fit_variations = []
    
    return fit_parameters_dict, fit_variations

# Function to initialize a plot
def initialize_plot(q_len):
    """
    Initialize a Matplotlib figure with specific subplots layout for later data plotting.

    Args:
        q_len (int): The number of curves to determine the colormap size.

    Returns:
        tuple: A tuple containing two figures with their respective subplots, a colormap, and a dictionary to organize legend lines and labels.
    """
    # Create a new figure for each HDF5 file with a 3x2 subplot grid
    fig1 = plt.figure(figsize=(20, 16))
    axs1 = fig1.subplots(2, 3)

    fig2 = plt.figure(figsize=(20, 16))
    axs2 = fig2.subplots(2, 3)

    # Set titles for each row of subplots
    #fig1.suptitle("Page 1", fontsize=16)
    #fig2.suptitle("Page 2", fontsize=16)

    # Set titles for each row of subplots
    fig1.text(0.5, 0.975, "Correlation function fitting", fontsize=18, ha='center', va='center')
    fig1.text(0.5, 0.475, "Single exponential", fontsize=18, ha='center', va='center')

    fig2.text(0.5, 0.975, "Stretched exponential", fontsize=18, ha='center', va='center')
    fig2.text(0.5, 0.475, "Cumulants model", fontsize=18, ha='center', va='center')

    # Access individual subplots on the first page
    ax_single = axs1[0, 0]
    ax_stretched = axs1[0, 1]
    ax_cumulants = axs1[0, 2]
    ax_Cq2_si = axs1[1, 0]
    ax_Cq_si = axs1[1, 1]
    ax_si = axs1[1, 2]
    ax_si.axis('off')

    # Access individual subplots on the second page
    ax_Cq2_st = axs2[0, 0]
    ax_Cq_st = axs2[0, 1]
    ax_gammaq = axs2[0, 2]
    ax_Cq2_cu = axs2[1, 0]
    ax_Cq_cu = axs2[1, 1]
    ax_PDIq = axs2[1, 2]

    # Set aspect ratio to 'equal' for square plots
    for ax in [ax_single, ax_stretched, ax_cumulants, ax_Cq2_si, ax_Cq_si, ax_Cq2_st, ax_Cq_st, ax_gammaq, ax_Cq2_cu, ax_Cq_cu, ax_PDIq]:
        ax.set_box_aspect(1)

    # Adjust subplot parameters to reduce margins
    fig1.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.25, hspace=0.25)
    fig2.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.25, hspace=0.25)

    # Initialize a color map for different q values
    cmap = plt.cm.get_cmap('gist_rainbow', q_len)

    # Initialize a dictionary to store lines and labels for legend items
    lines_labels_dict = {
        "exp_lines_single": [],
        "exp_labels_single": [],
        "fit_lines_single": [],
        "fit_labels_single": [],
        "exp_lines_stretched": [],
        "exp_labels_stretched": [],
        "fit_lines_stretched": [],
        "fit_labels_stretched": [],
        "exp_lines_cumulants": [],
        "exp_labels_cumulants": [],
        "fit_lines_cumulants": [],
        "fit_labels_cumulants": [],
        "exp_lines_selq": [],
        "exp_labels_selq": [],
        "fit_lines_selq": [],
        "fit_labels_selq": [],
    }

    return fig1, (ax_single, ax_stretched, ax_cumulants, ax_Cq2_si, ax_Cq_si), fig2, (ax_Cq2_st, ax_Cq_st, ax_gammaq, ax_Cq2_cu, ax_Cq_cu, ax_PDIq), cmap, lines_labels_dict

# Function to initialize empty data structures for derived and qualified parameters
def initialize_data_for_derived_params():
    """
    Initializes two dictionaries to store derived and qualified parameters from the XPCS analysis.

    Returns:
        tuple: A tuple containing two dictionaries:
            1. derived_params (dict): A dictionary to store derived parameters, including:
                - q_value: List to store the q values
                - q_square: List to store the squared q values
                - relax_rate_single: List to store the relaxation rates from the single exponential model
                - relax_rate_stretched: List to store the relaxation rates from the stretched exponential model
                - gamma: List to store the gamma (stretching exponent) values from the stretched exponential model
                - relax_rate_cumulants: List to store the relaxation rates from the cumulants method
                - PDI: List to store the polydispersity index from the cumulants method
            2. qualified_params (dict): A dictionary to store the qualified parameters, which are the parameters
               from the fits that have an R-squared (R2) value greater than the R2_THRESHOLD. This dictionary includes:
                - q_value: List to store the q values for qualified fits
                - relax_rate_single: List to store the relaxation rates from the single exponential model for qualified fits
                - relax_rate_stretched: List to store the relaxation rates from the stretched exponential model for qualified fits
                - relax_rate_cumulants: List to store the relaxation rates from the cumulants method for qualified fits
                - gamma: List to store the gammas from the stretched exponential model for qualified fits
                - PDI: List to store the PDIs from the cumulants method for qualified fits
    """

    # Initialize empty lists for derived parameters
    derived_params = {
        "q_value": [],
        "q_square": [],   
        "relax_rate_single": [],
        "relax_rate_stretched": [],
        "gamma": [],  
        "relax_rate_cumulants": [],
        "PDI": []
    }

    # Initialize empty lists for qualified parameters
    qualified_params = {
        "q_value_single": [],
        "q_value_stretched": [],
        "q_value_cumulants": [],
        "relax_rate_single": [],
        "relax_rate_stretched": [],
        "relax_rate_cumulants": [],
        "color_single": [],
        "color_stretched": [],
        "color_cumulants": [],
        "gamma": [],
        "PDI": []
    }
        
    return derived_params, qualified_params

#-----------------------------------------------------------------------#
#------ Interactive parameter selection block (for a single file) ------#
#-----------------------------------------------------------------------#

# Class for creating an interactive graph window
class GraphWindow:
    """
    Create an interactive graph window for q selection.

    Attributes:
        root (Tk): The root Tkinter window.
        new_dataset_df (DataFrame): The dataset for graphing.
        lower_limit (int): The lower limit for 't' range.
        upper_limit (int): The upper limit for 't' range.
        cmap (function): The color map function.
        figure (Figure): The Matplotlib figure for the graph.
        ax (Axes): The Matplotlib axes for the graph.
        canvas (FigureCanvasTkAgg): The Matplotlib canvas for Tkinter.
        selected_values (list): List to store selected q values.
        selected_columns (list): List to store selected q columns in DataFrame.
        lines_labels_dict (dict): Dictionary to store lines and labels information.
        q_buttons (list): List to store references to the q value selector checkboxes.
    """
    def __init__(self, root, new_dataset_df, lower_limit, upper_limit, cmap):
        """Initialize the GraphWindow."""
        # Initialize the root window
        self.root = root
        self.root.title("Select q values")

        # Initialize the selected values
        self.selected_values = []

        # Initialize the selected columns
        self.selected_columns = []

        # Initialize q_buttons list
        self.q_buttons = []

        # Store dataset information
        self.new_dataset_df = new_dataset_df
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.cmap = cmap

        # Create lines and labels dictionary
        self.lines_labels_dict = {
            "exp_lines_selq": [],
            "exp_labels_selq": [],
            "fit_lines_selq": [],
            "fit_labels_selq": []
        }

        # Create the interactive graph
        self.create_graph()

        # Create the q value selector checkboxes
        self.create_q_selector()

        # Create the Select and Exit buttons
        self.create_buttons()

        # Center the window on the screen
        self.root.update_idletasks()
        width = self.root.winfo_reqwidth()
        height = self.root.winfo_reqheight()
        x = (self.root.winfo_screenwidth() - width) // 2
        y = (self.root.winfo_screenheight() - height) // 2
        self.root.geometry("+%d+%d" % (x, y))

    # Method to create the interactive graph
    def create_graph(self):
        """Create the interactive graph."""
        self.figure = Figure(figsize=(8, 5))
        self.ax = self.figure.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=1)

        for i, q_column in enumerate(self.new_dataset_df.columns[1:], 1):

            match = re.search(r"q=(\d+\.\d+)", q_column)

            if not match:
                continue

            q_value = float(match.group(1))

            t_q = self.new_dataset_df['# t']
            g2_q = self.new_dataset_df[q_column]

            t_q = t_q.iloc[self.lower_limit:self.upper_limit + 1]
            g2_q = g2_q.iloc[self.lower_limit:self.upper_limit + 1]

            fit_params_selq, fitted_curve_selq, r2_selq = fit_single_exponential(t_q, g2_q)

            A_selq = fit_params_selq[0]
            B_selq = fit_params_selq[1]

            color = self.cmap(i - 1)

            line_exp_sq = self.ax.semilogx(
                t_q,
                (g2_q - A_selq) / B_selq,
                'o',
                color=color
            )[0]

            line_fit_sq = self.ax.semilogx(
                t_q,
                (fitted_curve_selq - A_selq) / B_selq,
                color=color,
                linestyle='-'
            )[0]

            self.lines_labels_dict["exp_lines_selq"].append(line_exp_sq)
            self.lines_labels_dict["exp_labels_selq"].append(f"q = {q_value:.6f}")
            self.lines_labels_dict["fit_lines_selq"].append(line_fit_sq)
            self.lines_labels_dict["fit_labels_selq"].append(f"R2: {r2_selq:.3f}")

        self.ax.set_title('Correlation function fitting - Single Exponential Model')
        self.ax.set_xlabel('Delay Time (s)')
        self.ax.set_ylabel(r'$(g_2 - \mathrm{baseline}) / \beta$')

        # Do not add a legend inside the plot.
        # The q and R2 labels are shown in the checkbox list below the graph.
        self.figure.tight_layout()

        self.canvas.draw()

    # Method to create the q value selector checkboxes
    def create_q_selector(self):
        """Create the q value selector checkboxes."""
        from matplotlib.colors import to_hex

        self.selected_values = []
        self.selected_columns = []
        self.q_buttons = []

        q_frame = ttk.Frame(self.root)
        q_frame.pack(side='top', fill='both', expand=True)

        q_canvas = tk.Canvas(q_frame)
        q_canvas.pack(side='left', fill='both', expand=True)

        q_scrollbar = ttk.Scrollbar(q_frame, orient="vertical", command=q_canvas.yview)
        q_scrollbar.pack(side='right', fill='y')

        inner_frame = ttk.Frame(q_canvas)
        q_canvas.create_window((0, 0), window=inner_frame, anchor='nw')

        for q_index, (q_column, exp_label, fit_label, exp_line) in enumerate(
            zip(
                self.new_dataset_df.columns[1:],
                self.lines_labels_dict["exp_labels_selq"],
                self.lines_labels_dict["fit_labels_selq"],
                self.lines_labels_dict["exp_lines_selq"]
            )
        ):

            match = re.search(r"q=(\d+\.\d+)", q_column)

            if not match:
                continue

            q_value = float(match.group(1))

            # All q values start selected because all curves start visible.
            self.selected_values.append(q_value)
            self.selected_columns.append(q_column)

            btn_var = tk.BooleanVar(value=True)

            # Use the same color as the corresponding curve.
            curve_color = to_hex(exp_line.get_color())

            row_frame = ttk.Frame(inner_frame)
            row_frame.pack(side='top', anchor='w', fill='x', padx=5, pady=2)

            btn = tk.Checkbutton(
                row_frame,
                variable=btn_var,
                command=lambda idx=q_index, var=btn_var: self.toggle_curve(idx, var),
                selectcolor=curve_color,
                anchor='w'
                # foreground=curve_color,          # Optional: use colored text instead of black text.
                # activeforeground=curve_color     # Optional: use colored active text instead of black text.
            )
            btn.pack(side='left')

            color_dot = tk.Label(
                row_frame,
                text="●",
                foreground=curve_color
            )
            color_dot.pack(side='left', padx=(2, 5))

            q_label = tk.Label(
                row_frame,
                text=f"{exp_label} - {fit_label}",
                foreground="black",
                anchor='w',
                justify='left'
            )
            q_label.pack(side='left')

            # Allow clicking on the dot or label to toggle the checkbox.
            color_dot.bind("<Button-1>", lambda event, button=btn: button.invoke())
            q_label.bind("<Button-1>", lambda event, button=btn: button.invoke())

            self.q_buttons.append((btn, btn_var))

        inner_frame.update_idletasks()

        q_canvas.config(scrollregion=q_canvas.bbox("all"))
        q_canvas.config(yscrollcommand=q_scrollbar.set)

        q_canvas.bind(
            '<Configure>',
            lambda event, q_canvas=q_canvas: q_canvas.configure(
                scrollregion=q_canvas.bbox("all")
            )
        )

    # Method to toggle q curve visibility
    def toggle_curve(self, q_index, var):
        """Toggle visibility of the curve for the specified q index."""
        is_visible = var.get()

        # Get the experimental and fitted curves associated with this q value.
        exp_line = self.lines_labels_dict["exp_lines_selq"][q_index]
        fit_line = self.lines_labels_dict["fit_lines_selq"][q_index]

        # Checkbox selected = curve visible.
        # Checkbox unselected = curve hidden.
        exp_line.set_visible(is_visible)
        fit_line.set_visible(is_visible)

        # Rebuild the selected q values and selected DataFrame columns.
        self.selected_values = []
        self.selected_columns = []

        for column, line in zip(
            self.new_dataset_df.columns[1:],
            self.lines_labels_dict["exp_lines_selq"]
        ):
            if line.get_visible():
                match = re.search(r"q=(\d+\.\d+)", column)

                if match:
                    self.selected_values.append(float(match.group(1)))
                    self.selected_columns.append(column)

        # Redraw the canvas to reflect the changes.
        self.canvas.draw_idle()

    def create_buttons(self):
        """Create the Select and Exit buttons."""
        ttk.Button(self.root, text="Select", command=self.select_values).pack(side='right', padx=10)
        ttk.Button(self.root, text="Exit", command=self.exit_window).pack(side='left', padx=10)

    def select_values(self):
        """Save the selected values and close the window."""
        #print("Selected values:", self.selected_values)

        # Filter the columns of the original DataFrame based on the selected q values
        selected_columns = ['# t'] + self.selected_columns
        filtered_dataset_df = self.new_dataset_df[selected_columns]

        # Modify the original DataFrame
        self.new_dataset_df = filtered_dataset_df

        # Close the Tkinter window
        self.root.quit()
        #self.root.destroy()

        # Return the filtered DataFrame
        return filtered_dataset_df

    def exit_window(self):
        """
        Close the q-selection window without applying additional filtering.

        The original DataFrame is kept unchanged. The Tkinter window is not
        destroyed here because the notebook destroys it after mainloop().
        """
        self.root.quit()

#-----------------------------------------------------------------------#
#----------------------------- Fit Models ------------------------------#
#-----------------------------------------------------------------------#

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

            # Restriction for parameters A B C X based on the model
            if model_func.__name__ == 'single_exponential':
                bounds = ([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf])  # A>0
            elif model_func.__name__ == 'stretched_exponential':
                bounds = ([0, -np.inf, -np.inf, 0], [np.inf, np.inf, np.inf, np.inf])  # A>0 gamma>0 
                #bounds = ([0, 0, -np.inf, 0], [2, 1, np.inf, 5]) 
            elif model_func.__name__ == 'stretched_exponential_refine':
                bounds = ([0, 0], [np.inf, np.inf])                            # C > 0, gamma > 0
                model_func.__name__ = 'stretched_exponential'
            elif model_func.__name__ == 'cumulants_model':
                bounds = ([0, -np.inf, -np.inf, 0], [np.inf, np.inf, np.inf, np.inf])  # A>0 C2>0
            elif model_func.__name__ == 'cumulants_model_refine':
                bounds = ([0, 0], [np.inf, np.inf])  # A>0 C2>0

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
    return A + B * np.exp(-2 * (C * t) ** gamma)

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

# Function to fit a single exponential model to the provided data
def fit_single_exponential(t, g2):
    """
    Fit a single exponential model to the provided data.

    Args:
        t (numpy.ndarray): Array of delay times.
        g2 (numpy.ndarray): Array of experimental g2 values.

    Returns:
        tuple: A tuple containing fit parameters, fitted curve, and R-squared value.
    """
    # Calculate the initial guess for parameters A and B:
    A0 = np.mean(g2[-5:])     # Average of the last 5 points
    B0 = np.mean(g2[:2]) - 1  # Average of the first 2 points minus 1

    # Calculate the initial guess for parameter C:
    # Obtain y
    y = (np.mean(g2[:2]) + np.mean(g2[-5:])) / 2
    # Find the closest x value in the experimental curve
    closest_index = np.abs(g2 - y).argmin()
    closest_x = t[closest_index]
    # Obtain C0:
    C0 = 1 / closest_x

    # Initial guess for parameters A, B, and C:
    initial_params_single = [A0, B0, C0]

    # Fit the curve g2 = A + B * exp(-2C * t)
    # With constraints
    fit_params_single, fitted_curve_single, r2_single = fit_model_with_constraints(t, g2, single_exponential, initial_params_single)
    # Without constraints
    #fit_params_single, fitted_curve_single, r2_single = fit_model(t, g2, single_exponential, initial_params_single)

    return fit_params_single, fitted_curve_single, r2_single

# Function to fit a stretched exponential model to the provided data
def fit_stretched_exponentialX(t, g2, fit_params_single):
    """
    Fit a stretched exponential model to the provided data.

    Args:
        t (numpy.ndarray): Array of delay times.
        g2 (numpy.ndarray): Array of experimental g2 values.
        fit_params_single (list): Parameters obtained from the single exponential fit.

    Returns:
        tuple: A tuple containing fit parameters, fitted curve, and R-squared value.
    """
    # Initial guess for parameter gamma
    gamma0 = 1

    # Use the parameters A, B, C from the Single Exponential fit and gamma0 as initial guess
    initial_params_stretched = [*fit_params_single, gamma0]

    # Fit the curve g2 = A + B * exp(-2C * t)**gamma
    # With constraints
    fit_params_stretched, fitted_curve_stretched, r2_stretched = fit_model_with_constraints(t, g2, stretched_exponential, initial_params_stretched)
    # Without constraints
    #fit_params_stretched, fitted_curve_stretched, r2_stretched = fit_model(t, g2, stretched_exponential, initial_params_stretched)

    return fit_params_stretched, fitted_curve_stretched, r2_stretched

# Function to fit a stretched exponential model to the provided data
def fit_stretched_exponential(t, g2, fit_params_single):
    """
    Refine the stretched exponential fit using fixed A and B from the single-exponential fit.
    In this step, only C and gamma are fitted, keeping A and B fixed.

    The model is:
        g2(t) = A + B * exp(-2 * (C * t)^gamma)

    Parameters:
        t (array-like): The time values.
        g2 (array-like): The experimental g2 data.
        fit_params_single (array-like): Parameters [A, B, C] obtained from the single exponential fit.

    Returns:
        tuple: (fit_params, fitted_curve, r2)
            fit_params: array-like, [A, B, C, gamma]
            fitted_curve: array-like, the fitted curve
            r2: float, R² score of the fit
    """
    # Extract A, B, C from the single_exponential fit
    A, B, C_single = fit_params_single

    # Initial guesses for C and gamma:
    C0 = C_single
    gamma0 = 1.0
    initial_params = [C0, gamma0]

    # Prepare the model with fixed A and B
    def stretched_exponential(t, C, gamma, A, B):
        return A + B * np.exp(-2 * (C * t) ** gamma)

    model_func = partial(stretched_exponential, A=A, B=B)
    model_func.__name__ = 'stretched_exponential_refine'  # For fit_model_with_constraints

    # Fit with constraints
    fit_params, fitted_curve, r2 = fit_model_with_constraints(t, g2, model_func, initial_params)

    # fit_params currently contains [C, gamma]
    # We want [A, B, C, gamma]
    if not np.isnan(fit_params).any():
        C_fit, gamma_fit = fit_params
        fit_params_full = [A, B, C_fit, gamma_fit]
    else:
        fit_params_full = [np.nan, np.nan, np.nan, np.nan]

    return fit_params_full, fitted_curve, r2

# Function to fit a cumulants model to the provided data
def fit_cumulantsX(t, g2, fit_params_single):
    """
    Fit a cumulants model to the provided data.

    Args:
        t (numpy.ndarray): Array of delay times.
        g2 (numpy.ndarray): Array of experimental g2 values.
        fit_params_single (list): Parameters obtained from the single exponential fit.

    Returns:
        tuple: A tuple containing fit parameters, fitted curve, and R-squared value.
    """
    # Initial guess for parameter C2
    C2_0 = 0.05*fit_params_single[2]**2

    # Use the parameters A, B, C from the Single Exponential fit and C2 as initial guess
    initial_params_cumulants = [*fit_params_single, C2_0]

    # Fit the curve g2 = A + B * exp(-2C1 * t)*(1 + (1/2)* C2 * t**2)**2
    # With constraints
    fit_params_cumulants, fitted_curve_cumulants, r2_cumulants = fit_model_with_constraints(t, g2, cumulants_model, initial_params_cumulants)
    # Without constraints
    #fit_params_cumulants, fitted_curve_cumulants, r2_cumulants = fit_model(t, g2, cumulants_model, initial_params_cumulants)
        
    return fit_params_cumulants, fitted_curve_cumulants, r2_cumulants

def fit_cumulants(t, g2, fit_params_single):
    """
    Fit a cumulants model to the provided data using fixed A and B from the single exponential fit.
    
    The model is:
        g2(t) = A + B * exp(-2 * C1 * t) * (1 + 0.5 * C2 * t^2)^2
    
    Parameters:
        t (array-like): Time values.
        g2 (array-like): Experimental g2 data.
        fit_params_single (array-like): Parameters [A, B, C1] from the single exponential fit.
    
    Returns:
        tuple: (fit_params_full, fitted_curve, r2)
            fit_params_full (list): [A, B, C1, C2]
            fitted_curve (array-like): The fitted g2(t) curve.
            r2 (float): Coefficient of determination.
    """
    # Extract A, B, and C1 from the single exponential fit
    A, B, C1_single = fit_params_single
    
    # Initial guesses for C1 and C2
    C1_0 = C1_single
    C2_0 = 0.05 * (C1_0)**2  
    initial_params = [C1_0, C2_0]

    # Prepare the model with fixed A and B
    def cumulants_model(t, C1, C2, A, B):
        return A + B * np.exp(-2 * C1 * t) * (1 + (1 / 2) * C2 * t**2)**2
    
    # Since A and B are fixed, we can use partial to fix these parameters
    model_func = partial(cumulants_model, A=A, B=B)
    model_func.__name__ = 'cumulants_model_refine'  # For fit_model_with_constraints

    # Fit with constraints
    fit_params, fitted_curve, r2 = fit_model_with_constraints(t, g2, model_func, initial_params)

    # fit_params contains [C1, C2]
    if not np.isnan(fit_params).any():
        C1_fit, C2_fit = fit_params
        fit_params_full = [A, B, C1_fit, C2_fit]
    else:
        fit_params_full = [np.nan, np.nan, np.nan, np.nan]
    
    return fit_params_full, fitted_curve, r2

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
    std_err (float): Standard error of the slope.
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

       # Get the standard error of the slope
       std_err = np.sqrt(np.diag(cov_matrix))[0]
       
       return D, pearson_r, std_err
   
    except (RuntimeError, ValueError):
        # Handle fitting errors, e.g., NaN or infinite values
        return np.nan, np.nan, np.nan
    
# Define the function that will perform the exponential fit and return the parameters
def fit_exponential_model(q_values, C_values):
    """
    Fit an exponential model C = D * q^n to given data.

    Parameters:
    q_values (array-like): Array of q values.
    C_values (array-like): Array of corresponding relaxation rates.

    Returns:
    D (float): Diffusion coefficient obtained from the exponential fit.
    n (float): Exponent obtained from the exponential fit.
    r2_score (float): Coefficient of determination (R^2) for the fit.
    std_err_D (float): Standard error of the coefficient D.
    std_err_n (float): Standard error of the exponent n.
    """
    # Convert q_values and C_values to float values
    q_values = np.array(q_values, dtype=np.float64)
    C_values = np.array(C_values, dtype=np.float64)

    # Define the exponential function to fit
    def exponential_function(q, D, n):
        return D * q**n

    try:
        # Perform the exponential fit
        fit_params, cov_matrix = curve_fit(exponential_function, q_values, C_values)

        # Get the parameters D and n
        D = fit_params[0]
        n = fit_params[1]
        #D, n = params
        
        # Calculate the predicted values
        predicted_values = D * q_values**n

        # Calculate the R^2 score
        r2 = r2_score(C_values, predicted_values)

        # Get the standard errors of D and n
        std_err_D, std_err_n = np.sqrt(np.diag(cov_matrix))

        return D, n, r2, std_err_D, std_err_n

    except (RuntimeError, ValueError):
        # Handle fitting errors, e.g., NaN or infinite values
        return np.nan, np.nan, np.nan, np.nan, np.nan

# Function to compare the code fit parameters with those of the beamline
def compare_fit_parameters(q_value, beamline_params, code_params):
    """
    Compares fit parameters from HDF5 data and code-generated data for the Stretched Exponential model.
    Calculates the percentage variation for each parameter.

    Args:
        q_value (float): The q-value being compared.
        beamline_params (dict): Dictionary containing HDF5 fit parameters.
        code_params (dict): Dictionary containing code-generated fit parameters.

    Returns:
        dict: Dictionary containing the percentage variations.
    """
    variation = {}
    try:
        variation['q_value'] = q_value
        # Calculate percentage variations for each parameter
        for param in ['Baseline', 'Beta', 'Relax_time', 'Gamma']:
            hdf5_value = beamline_params[param]
            code_value = code_params[param]
            if hdf5_value != 0:
                variation[f'{param}_variation_%'] = ((code_value - hdf5_value) / hdf5_value) * 100
            else:
                variation[f'{param}_variation_%'] = float('nan')  # Avoid division by zero
    except Exception as e:
       #print(f"### Error ###: Failed to calculate variation for q = {q_value} A^-1. Error: {e}")
        variation = {
            'q_value': q_value,
            'Baseline_variation_%': float('nan'),
            'Beta_variation_%': float('nan'),
            'Relax_time_variation_%': float('nan'),
            'Gamma_variation_%': float('nan')
        }
    return variation

##
def plot_fit_variations(variations_df):
    """
    Plots the percentage variations of fit parameters.

    Args:
        variations_df (pd.DataFrame): DataFrame containing percentage variations.
    """
    plt.figure(figsize=(12, 8))
    
    # Define the parameters to plot
    parameters = ['Baseline_variation_%', 'Beta_variation_%', 'Relax_time_variation_%', 'Gamma_variation_%']
    labels = ['Baseline', 'Beta', 'Relax Time', 'Gamma']
    markers = ['o', 's', '^', 'D']
    
    # Plot each parameter's variation
    for param, label, marker in zip(parameters, labels, markers):
        plt.plot(variations_df['q_value'], variations_df[param], marker=marker, linestyle='-', label=label)
    
    # Set plot labels and title
    plt.xlabel('q (A$^{-1}$)', fontsize=14)
    plt.ylabel('Percentage Variation (%)', fontsize=14)
    plt.title('Percentage Variation of Fit Parameters per q-value', fontsize=16)
    
    # Add legend and grid
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#-----------------------------------------------------------------------#
#------------------ Calculations and Data Processing -------------------#
#-----------------------------------------------------------------------#

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

#-----------------------------------------------------------------------#
#-------------------- Result Saving and Reporting ----------------------#
#-----------------------------------------------------------------------#

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
        fitted_curve_stretched = A_stretched + B_stretched * np.exp(-2 * (C_stretched * t) ** gamma)
        fitted_curve_cumulants = A_cumulants + B_cumulants * np.exp(-2 * C1_cumulants * t) * (1 + (1 / 2) * C2_cumulants * t**2)**2

        for i in range(len(t)):
            row_str = f"{t[i]}\t{g2[i]}\t{np.std(g2)}\t{fitted_curve_single[i]}\t{fitted_curve_stretched[i]}\t{fitted_curve_cumulants[i]}\n"
            dat_file.write(row_str)

# Function to print error and success counters
def print_summary(counters, directory=None, r2_threshold=None):
    """
    Print a summary of processing results.

    Parameters:
        counters (dict): A dictionary containing counters and lists for error,
                         success, and fit-quality tracking.
        directory (str, optional): Directory where the low-R2 report will be saved.
        r2_threshold (float, optional): R2 threshold used during analysis.
    """
    print(f"Total HDF5 files: {counters['hdf5_files']}")
    print(f"Total q values: {counters['q_values']}")
    print(f"Total invalid files: {counters['invalid_data']}")

    # Print the base_names of invalid files
    if counters['invalid_data'] != 0:
        print("Invalid files:")
        for file_name in counters['invalid_files']:
            print(file_name)

    counters['success'] = counters['q_values'] - counters['failure']

    print(f"Successful fits: {counters['success']}")
    print(f"Failed fits: {counters['failure']}")
    print(f"Fits below R2 threshold: {counters['below_threshold']}")

    # Print the base_names of failed files
    if counters['failure'] != 0:
        print("Failed fits files:")
        for file_name in counters['failed_files']:
            print(file_name)

    # Save low-quality fits to a separate report file instead of printing them all.
    if counters['below_threshold'] != 0:
        if directory is not None:
            below_threshold_path = os.path.join(
                directory,
                "below_R2_threshold_fits.dat"
            )

            with open(below_threshold_path, "w") as report_file:
                report_file.write("# Fits below R2 threshold\n")

                if r2_threshold is not None:
                    report_file.write(f"# R2_threshold = {r2_threshold}\n")

                report_file.write("# These fits converged numerically but were excluded from averages and diffusion-related plots.\n")
                report_file.write("# Format: filename - q - model - R2\n")

                for file_name in counters['below_threshold_files']:
                    report_file.write(f"{file_name}\n")

            print(f"Detailed low-R2 fit list saved to: {below_threshold_path}")

        else:
            print("Detailed low-R2 fit list was not saved because no output directory was provided.")

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

#-----------------------------------------------------------------------#
#--------------------- Plotting and Visualization ----------------------#
#-----------------------------------------------------------------------#

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

# Function to fit and plot a specified model
def fit_and_plot_model(q_values, relax_rate_values, ax, colors, label, model_type='linear'):
    """
    Fit a model to given data and generate a plot on a specified subplot.

    Invalid, NaN, infinite, or non-positive relaxation-rate values are ignored.
    If there are not enough valid points, the function writes a message in the
    subplot and returns NaN values instead of raising an error.

    Args:
        q_values (array-like): Array of q values.
        relax_rate_values (array-like): Array of corresponding relaxation rates.
        ax (matplotlib.axes.Axes): The subplot where the plot will be generated.
        colors (list): List of colors for each data point.
        label (str): Label for the plot.
        model_type (str, optional): Type of model to fit. Supported values are
                                    'linear' or 'exponential'. Defaults to 'linear'.

    Returns:
        D (float): Diffusion coefficient obtained from the fit.
        n (float): Exponent obtained from the fit.
        r_metric (float): Pearson r for linear fit or R2 for exponential fit.
        std_err (float or tuple): Standard error for the fit.
    """
    try:
        q_values = np.asarray(q_values, dtype=float)
        relax_rate_values = np.asarray(relax_rate_values, dtype=float)
        colors = list(colors)

        # Keep only physically meaningful and finite values.
        valid_mask = (
            np.isfinite(q_values)
            & np.isfinite(relax_rate_values)
            & (q_values > 0)
            & (relax_rate_values > 0)
        )

        q_values = q_values[valid_mask]
        relax_rate_values = relax_rate_values[valid_mask]
        colors = [color for color, keep in zip(colors, valid_mask) if keep]

        # Need at least two points for a linear fit and at least three for a
        # stable two-parameter exponential/power-law fit.
        min_points = 2 if model_type == 'linear' else 3

        if len(q_values) < min_points:
            ax.text(
                0.5,
                0.5,
                "Not enough valid fits",
                transform=ax.transAxes,
                ha='center',
                va='center',
                fontsize=12
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

            print(
                f"Skipping {model_type} model: "
                f"not enough valid points ({len(q_values)} available)."
            )

            return np.nan, np.nan, np.nan, np.nan

        if model_type == 'linear':
            # Use q_value**2 as the x-values for the linear fit.
            x_values = q_values**2

            D, r_metric, std_err = fit_linear_model(x_values, relax_rate_values)
            n = np.nan

        elif model_type == 'exponential':
            # Use q_value as the x-values for the exponential fit.
            x_values = q_values

            D, n, r_metric, std_err_D, std_err_n = fit_exponential_model(
                x_values,
                relax_rate_values
            )
            std_err = (std_err_D, std_err_n)

        else:
            raise ValueError("Invalid model_type. Supported values are 'linear' or 'exponential'.")

        # Plot the valid model data.
        for x, y, color in zip(x_values, relax_rate_values, colors):
            ax.plot(x, y, 'o', color=color, markersize=10)

        # If the fit failed internally, do not try to draw a fit line.
        if not np.isfinite(D):
            ax.text(
                0.5,
                0.5,
                "Fit failed",
                transform=ax.transAxes,
                ha='center',
                va='center',
                fontsize=12
            )
            ax.set_xlim(0, max(x_values) * 1.1)
            ax.set_ylim(0, max(relax_rate_values) * 1.1)

            return D, n, r_metric, std_err

        # Define the exponential function for fitting.
        def exponential_function(q, D, n):
            return D * q**n

        # Generate the fit line using the fitted values.
        if model_type == 'linear':
            q_fit = np.linspace(0, max(x_values), 100)
            fit_line = D * q_fit

        elif model_type == 'exponential':
            q_fit = np.linspace(min(x_values), max(x_values), 100)
            fit_line = exponential_function(q_fit, D, n)

        # Plot the fit line only if it is finite.
        if np.all(np.isfinite(fit_line)):
            ax.plot(
                q_fit,
                fit_line,
                linestyle='-',
                color='black',
                label=f"{model_type.capitalize()} Fit"
            )

        # Set axes safely.
        x_max = max(x_values)
        y_max = max(relax_rate_values)

        if np.isfinite(x_max) and x_max > 0:
            ax.set_xlim(0, x_max * 1.1)
        else:
            ax.set_xlim(0, 1)

        if np.isfinite(y_max) and y_max > 0:
            ax.set_ylim(0, y_max * 1.1)
        else:
            ax.set_ylim(0, 1)

        return D, n, r_metric, std_err

    except Exception as e:
        print(f"Error fitting and plotting the {model_type} model: {e}")

        ax.text(
            0.5,
            0.5,
            "Fit/plot error",
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=12
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        return np.nan, np.nan, np.nan, np.nan

# Function to plot a specified param
def plot_params(q_values, y_values, ax, cmap, param_type):
    """
    Plot points of a derived parameter on a specified subplot.

    Invalid, NaN, or infinite values are ignored. If no valid values remain,
    a message is written in the subplot instead of raising an error.

    Args:
        q_values (array-like): Array of q values.
        y_values (array-like): Array of parameter values.
        ax (matplotlib.axes.Axes): The subplot where the plot will be generated.
        cmap (matplotlib.colors.Colormap): Colormap for point colors.
        param_type (str): Type of parameter ('gamma' or 'PDI').

    Returns:
        None
    """
    try:
        q_values = np.asarray(q_values, dtype=float)
        y_values = np.asarray(y_values, dtype=float)

        valid_mask = (
            np.isfinite(q_values)
            & np.isfinite(y_values)
            & (q_values > 0)
        )

        q_values = q_values[valid_mask]
        y_values = y_values[valid_mask]

        if len(q_values) == 0:
            ax.text(
                0.5,
                0.5,
                f"No valid {param_type} values",
                transform=ax.transAxes,
                ha='center',
                va='center',
                fontsize=12
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return

        # Plot the valid data points.
        for i in range(len(q_values)):
            color = cmap(i)
            ax.plot(q_values[i], y_values[i], 'o', color=color, markersize=10)

        y_min = np.min(y_values)
        y_max = np.max(y_values)

        if not np.isfinite(y_min) or not np.isfinite(y_max):
            ax.set_ylim(0, 1)
            return

        # Set y-axis limits with a margin.
        if param_type.lower() == 'gamma':
            y_margin = max(0.01, 0.1 * (y_max - y_min))

            if y_max == y_min:
                y_min -= y_margin
                y_max += y_margin
            else:
                y_min -= y_margin
                y_max += y_margin

            ax.set_ylim(y_min, y_max)

        elif param_type.lower() == 'pdi':
            y_margin = max(0.01, 0.1 * (y_max - y_min))

            if y_max == y_min:
                y_min -= y_margin
                y_max += y_margin
            else:
                y_min -= y_margin
                y_max += y_margin

            ax.set_ylim(y_min, y_max)

    except Exception as e:
        print(f"Error plotting {param_type} points: {e}")

        ax.text(
            0.5,
            0.5,
            f"{param_type} plot error",
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=12
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
# Configure subplot with title, labels, experimental data legend, and fitted curves legend
# Function to configure subplot with title, labels, and legend
def configure_subplot(ax, title, xlabel, ylabel, exp_lines, exp_labels, fit_lines, fit_labels):
    """
    Configures a subplot with title, axis labels, and a combined legend.

    Parameters:
        ax (matplotlib.axes.Axes): The subplot to configure.
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
    ax.set_ylabel(ylabel.replace("Beta", "β"), fontsize=14)

    # Combine q values and R2 values in a single legend entry
    combined_labels = [
        f"{exp_label} - {fit_label}"
        for exp_label, fit_label in zip(exp_labels, fit_labels)
    ]

    # Let Matplotlib choose the least obstructive position.
    # This is better than forcing the legend to the upper-right corner.
    if combined_labels:
        legend_exp = ax.legend(
            exp_lines,
            combined_labels,
            loc='best',
            fontsize=8,
            frameon=True,
            framealpha=0.85,
            borderaxespad=0.5
        )
        ax.add_artist(legend_exp)

# Function to calculate the text position on the plot
def calculate_text_position(q_count):
    """
    Calculate text positions based on q_count.

    Parameters:
        q_count (int): The value of q_count.

    Returns:
        float: The text position for Diff Coef.
    """
    # Ajusta estos valores según la relación deseada
    slope = -0.043
    intercept = 1.0

    # Calcula la posición de diff_coef
    diff_coef_position = slope * q_count + intercept

    return diff_coef_position

# Function to add text at the least obstructive position
def add_text_best_position(ax, text, fontsize=12):
    """
    Add text to an axes object using a simple automatic placement algorithm.

    The function tests several candidate positions in axes coordinates and
    selects the one with the fewest plotted data points nearby. If a legend
    already exists, positions overlapping with the legend are penalized.

    Parameters:
        ax (matplotlib.axes.Axes): Axes object where the text will be added.
        text (str): Text to add.
        fontsize (int): Font size for the text.

    Returns:
        matplotlib.text.Text: The added text object.
    """
    if text is None or text == "":
        return None

    # Candidate positions in axes coordinates.
    # Each entry is: (x, y, horizontal alignment, vertical alignment).
    candidates = [
        (0.03, 0.97, 'left',   'top'),
        (0.97, 0.97, 'right',  'top'),
        (0.03, 0.50, 'left',   'center'),
        (0.97, 0.50, 'right',  'center'),
        (0.50, 0.97, 'center', 'top'),
        (0.50, 0.03, 'center', 'bottom'),
        (0.03, 0.03, 'left',   'bottom'),
        (0.97, 0.03, 'right',  'bottom'),
    ]

    # Approximate text-box size in axes coordinates.
    n_lines = text.count("\n") + 1
    max_line_length = max(len(line) for line in text.split("\n"))

    box_width = min(0.60, max(0.22, 0.010 * max_line_length))
    box_height = min(0.35, max(0.08, 0.075 * n_lines))

    # Collect plotted data in axes coordinates.
    points_axes = []

    for line in ax.lines:
        x_data = np.asarray(line.get_xdata(), dtype=float)
        y_data = np.asarray(line.get_ydata(), dtype=float)

        valid_mask = np.isfinite(x_data) & np.isfinite(y_data)

        if valid_mask.sum() == 0:
            continue

        xy_display = ax.transData.transform(
            np.column_stack([x_data[valid_mask], y_data[valid_mask]])
        )
        xy_axes = ax.transAxes.inverted().transform(xy_display)

        inside_mask = (
            (xy_axes[:, 0] >= 0) & (xy_axes[:, 0] <= 1) &
            (xy_axes[:, 1] >= 0) & (xy_axes[:, 1] <= 1)
        )

        points_axes.append(xy_axes[inside_mask])

    if points_axes:
        points_axes = np.vstack(points_axes)
    else:
        points_axes = np.empty((0, 2))

    # Get legend box in axes coordinates, if a legend already exists.
    legend_box = None
    legend = ax.get_legend()

    if legend is not None:
        try:
            ax.figure.canvas.draw()
            renderer = ax.figure.canvas.get_renderer()
            legend_bbox_display = legend.get_window_extent(renderer=renderer)
            legend_bbox_axes = ax.transAxes.inverted().transform(
                legend_bbox_display.get_points()
            )

            legend_box = (
                legend_bbox_axes[0, 0],
                legend_bbox_axes[1, 0],
                legend_bbox_axes[0, 1],
                legend_bbox_axes[1, 1]
            )
        except Exception:
            legend_box = None

    def get_box_limits(x, y, ha, va):
        """Return approximate text-box limits in axes coordinates."""
        if ha == 'left':
            x0, x1 = x, x + box_width
        elif ha == 'right':
            x0, x1 = x - box_width, x
        else:
            x0, x1 = x - box_width / 2, x + box_width / 2

        if va == 'bottom':
            y0, y1 = y, y + box_height
        elif va == 'top':
            y0, y1 = y - box_height, y
        else:
            y0, y1 = y - box_height / 2, y + box_height / 2

        return x0, x1, y0, y1

    def boxes_overlap(box_a, box_b):
        """Check whether two boxes overlap."""
        if box_a is None or box_b is None:
            return False

        ax0, ax1, ay0, ay1 = box_a
        bx0, bx1, by0, by1 = box_b

        return not (
            ax1 < bx0 or
            ax0 > bx1 or
            ay1 < by0 or
            ay0 > by1
        )

    best_candidate = candidates[0]
    best_score = np.inf

    for candidate in candidates:
        x, y, ha, va = candidate
        text_box = get_box_limits(x, y, ha, va)
        x0, x1, y0, y1 = text_box

        # Penalize boxes that go outside the plot.
        outside_penalty = 0
        if x0 < 0 or x1 > 1 or y0 < 0 or y1 > 1:
            outside_penalty = 1000

        # Penalize overlap with plotted points.
        if points_axes.size == 0:
            point_overlap_score = 0
        else:
            overlap_mask = (
                (points_axes[:, 0] >= x0) & (points_axes[:, 0] <= x1) &
                (points_axes[:, 1] >= y0) & (points_axes[:, 1] <= y1)
            )
            point_overlap_score = overlap_mask.sum()

        # Penalize overlap with the legend much more strongly.
        legend_penalty = 0
        if boxes_overlap(text_box, legend_box):
            legend_penalty = 10000

        score = point_overlap_score + outside_penalty + legend_penalty

        if score < best_score:
            best_score = score
            best_candidate = candidate

    x, y, ha, va = best_candidate

    return ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        va=va,
        ha=ha,
        fontsize=fontsize,
        bbox=dict(
            facecolor='white',
            edgecolor='none',
            alpha=0.65,
            pad=1.5
        )
    )
    
