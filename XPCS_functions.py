# XPCS_functions.py - Module containing essential functions for XPCS data analysis.

# Author: Marilina Cathcarth [mcathcarth@gmail.com]
# Version: 7.0
# Date: October, 2026

import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import os
import tempfile
import h5py
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import warnings
from scipy.stats import pearsonr
from datetime import datetime, timezone
import textwrap
import errno

# Software version recorded in analysis outputs.
__version__ = '7.0'

#-----------------------------------------------------------------------#
#--------------------- Directory and File Handling ---------------------#
#-----------------------------------------------------------------------#

# Controlled stop for notebook execution
class AnalysisStopped(Exception):
    """
    Stop analysis with a readable message.

    Notes:
        IPython displays the message without a technical traceback.
    """

    def _render_traceback_(self):
        """Return the message as a list of strings for IPython."""
        return [str(self)]

# Function to prepare the output directory
def prepare_output_directory(directory, out_dir=''):
    """
    Create and check the output directory.

    Parameters:
        directory (str or os.PathLike): Input directory.
        out_dir (str or os.PathLike): Requested output directory.
            Empty: use the input directory.

    Returns:
        str: Output directory path.

    Raises:
        AnalysisStopped: Directory cannot be created or written.
    """
    # Use the requested output directory, or the input directory.
    output_directory = os.fspath(out_dir or directory)

    try:
        # Create missing directories; keep existing directories.
        os.makedirs(output_directory, exist_ok=True)

        # Check file creation in the destination.
        # The temporary file is removed automatically on closing.
        with tempfile.TemporaryFile(dir=output_directory):
            pass

    except OSError as error:
        # Display a custom message with the original technical details.
        raise AnalysisStopped(
            f"Could not prepare output directory '{output_directory}'. "
            f"Reason: {error}"
        ) from None

    return output_directory

# Function to create a separate summary directory for each execution
def prepare_summary_directory(output_directory):
    """
    Create a numbered directory for this execution's combined reports.

    Parameters:
        output_directory (str or os.PathLike): Main output directory.

    Returns:
        str: New '_summary/run_001', 'run_002', ... directory path.

    Raises:
        AnalysisStopped: Summary directory cannot be created.

    Notes:
        Existing directories and their reports are preserved.
        Each execution receives a new directory, including single-file runs.
    """
    summary_root = os.path.join(output_directory, '_summary')
    run_number = 1

    while True:
        summary_directory = os.path.join(summary_root, f'run_{run_number:03d}')

        try:
            # Require a new directory to protect earlier reports.
            os.makedirs(summary_directory, exist_ok=False)
            return summary_directory

        except FileExistsError:
            # Try the next number when this name is already in use.
            run_number += 1

        except OSError as error:
            raise AnalysisStopped(
                f"Could not create summary directory '{summary_directory}'. "
                f"Reason: {error}"
            ) from None

# Function to select synchrotron
def select_synchrotron():
    """
    Select a synchrotron data format. Display Tkinter selection windows.

    Returns:
        tuple: (synchrotron, data_type).
            - synchrotron (str): Selected input format.
            - data_type (str or None): Selected data representation.
    
    Raises:
        AnalysisStopped: Selection cancelled.

    Notes:
        Formats: 'Sirius 1', 'Sirius 2', 'Sirius 3', 'Sirius 3 average', 
        'APS', 'ESRF'.
        Data selection: Sirius 3 formats only.
        'Originals': original data.
        'Rebinneds': stored multi-tau data.
        Other formats: data_type = None.
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
        sirius_x_position = (screen_width - sirius_window_width) // 2
        sirius_y_position = (screen_height - sirius_window_height) // 2

        # Position is measured from the window's top-left corner.
        sirius_window.geometry(
            f"{sirius_window_width}x{sirius_window_height}"
            f"+{sirius_x_position}+{sirius_y_position}"
        )
        # Use the same geometry as the root window
        #sirius_window.geometry(root.geometry())

        # Create the label widget to display the instruction
        label = tk.Label(sirius_window, text="Select the Sirius version:")
        label.pack(pady=10)                 # Add some padding

        # Create the buttons to select Sirius versions
        sirius1_button = tk.Button(
            sirius_window, text="version 1 (May, 2023)", command=set_sirius_1)
        sirius1_button.pack(pady=5)

        sirius2_button = tk.Button(
            sirius_window, text="version 2 (October, 2023)", command=set_sirius_2)
        sirius2_button.pack(pady=5)

        sirius3_button = tk.Button(
            sirius_window, text="version 3 (January, 2024)", command=set_sirius_3)
        sirius3_button.pack(pady=5)

        sirius3m_button = tk.Button(
            sirius_window, text="version 3 (Average)", command=set_sirius_3m)
        sirius3m_button.pack(pady=5)

        # Bind the closing of the window to the on_closing function
        sirius_window.protocol("WM_DELETE_WINDOW", on_closing)

    def select_data_type(selected_type):
        # Modify the variable defined in the enclosing function.
        nonlocal data_type
        data_type = selected_type
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

        data_type_window.geometry(
            f"{data_type_window_width}x{data_type_window_height}"
            f"+{data_type_x_position}+{data_type_y_position}")

        # Create a label widget to display the instruction
        label = tk.Label(data_type_window, text="Select data type:")
        label.pack(pady=10)                 # Add some padding

        # Create the buttons to select data type
        # Keep the identifiers expected by the HDF5 readers.
        originals_button = tk.Button(data_type_window, text="Original Data", 
                                     command=lambda: select_data_type("Originals"))
        originals_button.pack(pady=5)

        rebineds_button = tk.Button(data_type_window, text="Multi-tau Data", 
                                    command=lambda: select_data_type("Rebinneds"))
                                    # Keep the identifier expected by the HDF5 readers.
        rebineds_button.pack(pady=5)

        # Treat closing this window as cancellation of the selection.
        data_type_window.protocol("WM_DELETE_WINDOW", on_closing)

    def set_sirius_1():
        nonlocal synchrotron
        synchrotron = 'Sirius 1'
        root.destroy()

    def set_sirius_2():
        nonlocal synchrotron
        synchrotron = 'Sirius 2'
        root.destroy()

    def set_sirius_3():
        nonlocal synchrotron
        synchrotron = 'Sirius 3'
        open_data_type_selection()
        #root.destroy()

    def set_sirius_3m():
        nonlocal synchrotron
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
        nonlocal synchrotron, data_type
        synchrotron = None
        data_type = None
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

    # Stop the analysis when the selection window is closed without a choice.
    if synchrotron is None:
        raise AnalysisStopped("Synchrotron selection cancelled.")

    return synchrotron, data_type

# Function to list HDF5 files in a directory
def list_hdf5_files(directory, parent=None):
    """
    List HDF5 files in a directory.

    Parameters:
        directory (str): Directory path.
        parent (tk.Misc or None, optional): Parent window for error messages.

    Returns:
        list or None: Sorted HDF5 filenames.
            []: No matching files.
            None: Directory could not be read.

    Notes:
        Regular .hdf5 and .h5 files; case-insensitive extensions.
        No subdirectory search. Filenames only, without directory paths.
        Read errors: display a message with technical details.
    """
    try:
        # Collect HDF5 files at this directory level.
        # Sort filenames; match extensions regardless of case.
        # Exclude directories with HDF5-like names.
        return sorted(
            filename
            for filename in os.listdir(directory)
            if filename.lower().endswith((".hdf5", ".h5"))
            and os.path.isfile(os.path.join(directory, filename))
        )
    except OSError as error:
        messagebox.showerror(
            "Directory Error",
            f"Could not read the directory:\n{error}",
            parent=parent
        )
        return None

# Function to select a directory or files
def select_directory_or_files():
    """
    Select a directory or individual HDF5 files from one directory.
    Display a Tkinter selection window.

    Returns:
        tuple: (directory_path, selected_files).
            - directory_path (str): Path to the selected directory.
            - selected_files (list): Selected HDF5 filenames.
            Cancelling returns ('', []).

    Notes:
        Directory selection: regular .hdf5 and .h5 files, sorted by name.
        No subdirectory search. 
        File selection: one directory; dialog order preserved.
        The caller joins each filename to directory_path.
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
        """Select a directory and collect its HDF5 files."""
        nonlocal directory_path, selected_files

        selected_path = filedialog.askdirectory(
            parent=root,
            title="Select Directory"
        )

        if selected_path:
            # Collect HDF5 files at this directory level.
            filenames = list_hdf5_files(selected_path, parent=root)

            # Keep the selection window open after a directory read error.
            if filenames is None:
                return

            # Store the selection after a successful directory read.
            directory_path = selected_path
            selected_files = filenames
        else:
            # Return an empty selection when the dialog is cancelled.
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
            # Normalize paths without changing the selection order.
            paths = [os.path.abspath(path) for path in selected_file_paths]

            # The 'All files' option also allows non-HDF5 selections.
            if any(
                not path.lower().endswith((".hdf5", ".h5"))
                or not os.path.isfile(path)
                for path in paths
            ):
                messagebox.showerror(
                    "File Selection Error",
                    "Select existing files with a .hdf5 or .h5 extension.",
                    parent=root
                )
                return

            # One directory is returned for all selected filenames.
            selected_directory = os.path.dirname(paths[0])
            if any(
                os.path.normcase(os.path.dirname(path))
                != os.path.normcase(selected_directory)
                for path in paths
            ):
                messagebox.showerror(
                    "File Selection Error",
                    "Select files from the same directory.",
                    parent=root
                )
                return

            directory_path = selected_directory
            selected_files = [os.path.basename(path) for path in paths]
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
    Ask whether to select a delay-time interval for fitting.

    Returns:
        bool: True for Yes; False for No or a closed dialog.

    Notes:
        This function only requests a choice.
        Interval limits: selected later by get_t_range_limits.
    """
    root = tk.Tk()
    root.withdraw()

    # Destroy the hidden window after the dialog, including on errors.
    try:
        user_response = messagebox.askyesno(
            "Define Range",
            "Do you want to define the range of 't'?",
            parent=root
        )
    finally:
        root.destroy()

    return user_response

# Function to generate the base name
def generate_base_name(selected_synchrotron, hdf5_file):
    """
    Generate the output base name.

    Parameters:
        selected_synchrotron (str): Input format.
        hdf5_file (str): Input filename or path.

    Returns:
        str: Output base name.

    Raises:
        ValueError: Unsupported input format.

    Notes:
        Final extension: removed for all formats.
        Sirius 1: remove the '_saxs_0000_RESULTS' suffix.
        Sirius 2 and 3: remove the two fields before the final field.
        Fewer than four underscore-separated fields: keep the full stem.
        Other formats: keep the full stem.
    """
    # Extract the filename and remove its final extension.
    filename = os.path.basename(hdf5_file)
    stem = os.path.splitext(filename)[0]

    if selected_synchrotron == 'Sirius 1':
        suffix = '_saxs_0000_RESULTS'
        if stem.endswith(suffix):
            base_name = stem[:-len(suffix)]
        else:
            base_name = stem

    elif selected_synchrotron in ('Sirius 2', 'Sirius 3'):
        parts = stem.split('_')

        # Keep the existing naming convention for these formats.
        if len(parts) >= 4:
            base_name = '_'.join(parts[:-3] + [parts[-1]])
        else:
            base_name = stem

    elif selected_synchrotron in ('Sirius 3 average', 'APS', 'ESRF'):
        base_name = stem

    else:
        raise ValueError(
            f"Unsupported input format: {selected_synchrotron!r}"
        )

    return base_name

# Function to prepare distinct output base names
def prepare_output_names(selected_synchrotron, hdf5_files):
    """
    Prepare distinct output base names for the selected files.

    Parameters:
        selected_synchrotron (str): Input format.
        hdf5_files (list): Selected input filenames.

    Returns:
        dict: Input filenames (str) mapped to output base names (str).

    Raises:
        AnalysisStopped: Unsupported format or conflicting output names.

    Notes:
        Reuse the naming rules from generate_base_name.
        Compare names case-insensitively for Windows compatibility.
        Check all selected filenames before processing.
    """
    output_names = {}
    owners = {}

    for filename in hdf5_files:
        try:
            # Generate the base name using the existing naming rules.
            base_name = generate_base_name(selected_synchrotron, filename)

        except ValueError as error:
            # Show the format error with the controlled notebook message.
            raise AnalysisStopped(str(error)) from None

        # Detect conflicts even when names differ only in letter case.
        name_key = base_name.casefold()

        if name_key in owners:
            raise AnalysisStopped(
                f"Output name conflict between '{owners[name_key]}' "
                f"and '{filename}'. "
                f"Their output base names match ignoring letter case: "
                f"'{base_name}'. "
                "Select these files separately or revise the output naming rule."
            )

        # Remember which input file produced this output name.
        owners[name_key] = filename

        # Store the prepared name for use in the processing loop.
        output_names[filename] = base_name

    return output_names

# Function to generate alphabetic output suffixes
def generate_alphabetic_suffix(index):
    """
    Generate an alphabetic suffix from a one-based index.

    Parameters:
        index (int): Position, starting at 1.

    Returns:
        str: 'a', ..., 'z', 'aa', 'ab', ...

    Raises:
        ValueError: Index smaller than 1.
    """
    if index < 1:
        raise ValueError("The suffix index must be at least 1.")

    # Continue with aa, ab, ... after z.
    suffix = ''
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        suffix = chr(ord('a') + remainder) + suffix

    return suffix

#-----------------------------------------------------------------------#
#---------------------------- HDF5 Readers -----------------------------#
#-----------------------------------------------------------------------#

# Function to process 'Multi-tau' data for Sirius
def process_multi_tau_sirius1(hdf):
    """
    Access the Sirius 1 Multi-tau group.

    Parameters:
        hdf (h5py.File): Open HDF5 file.

    Returns:
        tuple: (multi_tau, dataset_keys).
            - multi_tau (h5py.Group or None): Multi-tau group.
            - dataset_keys (list or Exception): Dataset names or read error.

    Notes:
        Failure: (None, error).
        The file must remain open while the group is used.
    """
    try:
        # Check if 'Multi-tau' key exists in HDF5 file
        multi_tau = hdf['Multi-tau']

        # Check that the object contains named datasets.
        if not isinstance(multi_tau, h5py.Group):
            raise ValueError("'Multi-tau' must be an HDF5 group.")
        
        # Read the keys of the datasets within Multi-tau
        dataset_keys = list(multi_tau.keys())

    # Handle exceptions that may occur during data processing
    except (OSError, KeyError, ValueError, IndexError) as e:
        return None, e
    
    return multi_tau, dataset_keys

# Function to process data from sirius_v1
def process_sirius1_data(file_path):
    """
    Read Sirius 1 correlation curves.

    Parameters:
        file_path (str): HDF5 file path.

    Returns:
        pd.DataFrame or None: Columns '# t' and 'g2(q=...)'.
            None: read failure or no usable curves.

    Notes:
        Source: 'Multi-tau' structured datasets.
        Fields: 'delay time (s)', 'g2'.
        Curves containing NaN or infinity: skipped.
        First sample: removed from each accepted curve.
        Accepted curves must share the same delay-time axis.
    """
    try:
        # Open the HDF5 file
        with h5py.File(file_path, 'r') as hdf:
            # Get the dataset keys and multi-tau data
            multi_tau, dataset_keys = process_multi_tau_sirius1(hdf)

            # Handle exceptions that may occur during data processing
            if multi_tau is None:
                error_message = (
                    "### Error ###:\n"
                    "The file does not belong to 'Sirius version 1' "
                    "or there is an issue with the data.\n"
                    f"{dataset_keys}"
                )
                print(error_message)
                return None
            
            # Initialize an empty DataFrame and a shared time axis
            combined_df = pd.DataFrame()
            t_reference = None
            
            # Iterate over the dataset keys
            for i, dataset_key in enumerate(dataset_keys):
                # Remove leading and trailing whitespace
                dataset_name = dataset_key.strip()
                
                # Extract data from the dataset within Multi-tau
                dataset = multi_tau.get(dataset_key)
                #if dataset is None:
                if not isinstance(dataset, h5py.Dataset):
                    continue

                # Require one dimension, with named fields for time and g2.
                if dataset.ndim != 1:
                    continue

                column = dataset[:]

                # Check that both required fields are present.
                if column.dtype.names is None or not {
                    'delay time (s)', 'g2'
                }.issubset(column.dtype.names):
                    continue
                
                # Get the values of the columns
                t = column['delay time (s)']
                g2 = column['g2']
                
                # Check for invalid values in the data using a boolean mask
                invalid_mask = (
                    np.isnan(t)
                    | np.isnan(g2)
                    | np.isinf(t)
                    | np.isinf(g2)
                )
                if invalid_mask.any():
                    continue
                
                # Remove the first data point
                t = t[1:]
                g2 = g2[1:]

                # Skip curves with no remaining samples.
                if t.size == 0:
                    continue
                
                # Get the 'q' value from the dataset name
                match = re.search(r"q\s*=\s*(\S+)", dataset_name)
                try:
                    q_value = float(match.group(1)) if match else np.nan
                except ValueError:
                    q_value = np.nan

                # Require a finite q value for the column name.
                if not np.isfinite(q_value):
                    print(
                        f"Skipping dataset {dataset_key!r}: invalid q value."
                    )
                    continue

                # Associate every accepted curve with the same time axis.
                if t_reference is None:
                    t_reference = t.copy()
                elif not np.array_equal(t, t_reference):
                    raise ValueError(
                        f"Dataset {dataset_key!r} has a different "
                        "delay-time axis."
                    )
                    
                # Add 'g2' data as a column in the DataFrame with the name 'g2(q=X)'
                combined_df[f"g2(q={q_value})"] = g2

            # Require at least one accepted curve.
            if t_reference is None:
                raise ValueError(
                    "No usable Sirius 1 correlation curves found."
                )
                
            # Insert the 't' column as the first column (column 0)
            combined_df.insert(0, '# t', t_reference)
            
        return combined_df

    # Handle exceptions that may occur during data processing
    except (OSError, KeyError, ValueError, IndexError) as e:
        error_message = (
            "### Error ###:\n"
            "The file does not belong to 'Sirius version 1' "
            "or there is an issue with the data.\n"
            f"{e}"
        )
        print(error_message)
        return None
    
# Function to process data from sirius_v2
def process_sirius2_data(file_path):
    """
    Read Sirius 2 one-time and two-time correlations.

    Parameters:
        file_path (str): HDF5 file path.

    Returns:
        tuple: (combined_df, t1t2_dataframes).
            - combined_df (pd.DataFrame or None): '# t', 'g2(q=...)'.
            - t1t2_dataframes (dict or None): q -> two-time DataFrame.
            Failure: (None, None).

    Notes:
        Two-time matrices keep their stored orientation.
        DataFrame row and column labels are integer indices.
    """
    try:
        with h5py.File(file_path, 'r') as hdf:
            ### q values ###
            # Navigate to the 'Fitting results' directory
            fitting_results = hdf['Fitting results']

            # Get the 'q' values from the dataset 'q values (angstroms)'
            q_values = fitting_results['q values (angstroms)'][:]

            ### t data ###
            # Navigate to the 'One-time correlation function' directory
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

            ### Two-time correlation ###
            # Navigate to the 'Two-time correlation' directory
            two_time_corr_function = hdf['Two-time correlation function']['2-TCF']

            # Initialize dictionary to store t1 vs t2 DataFrames for each q
            t1t2_dataframes = {}

            # Require one two-time matrix per q value.
            if len(two_time_corr_function) != len(q_values):
                raise ValueError(
                    'The number of two-time matrices does not match q.'
                )

            # Iterate over the datasets within the group and the corresponding q values
            for q, t1t2_dataset in zip(q_values, two_time_corr_function):
                # Convert each two-time matrix into a pandas DataFrame
                t1t2_df = pd.DataFrame(t1t2_dataset[:])
                # Add the DataFrame to the dictionary with q as the key
                t1t2_dataframes[q] = t1t2_df

        return combined_df, t1t2_dataframes
    
    # Handle exceptions that may occur during data processing
    except (OSError, KeyError, ValueError, IndexError) as e:
        error_message = (
                        "### Error ###:\n"
                        "The file does not belong to 'Sirius version 2' "
                        "or there is an issue with the data.\n"
                        f"{e}"
                    )
        print(error_message)
        return None, None
               
# Function to process data from sirius_v3
def process_sirius3_data(file_path,data_type):
    """
    Read Sirius 3 one-time and two-time correlations.

    Parameters:
        file_path (str): HDF5 file path.
        data_type (str): 'Originals' or 'Rebinneds'.

    Returns:
        tuple: (combined_df, t1t2_dataframes).
            - combined_df (pd.DataFrame or None): '# t', 'g2(q=...)'.
            - t1t2_dataframes (dict or None): q -> two-time DataFrame.
            Failure: (None, None).

    Notes:
        q values: 'Multi-tau Fitting results'.
        Two-time matrices keep their stored orientation and integer labels.
    """
    try:
        # Validate the selected data representation before reading.
        if data_type not in ('Originals', 'Rebinneds'):
            raise ValueError(
                "data_type must be 'Originals' or 'Rebinneds'."
            )
        
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

            ### Two-time correlation ###
            # Navigate to the 'Two-time correlation' directory
            two_time_corr_function = hdf['Two-time correlation']['2-TCF']

            # Initialize dictionary to store t1 vs t2 DataFrames for each q
            t1t2_dataframes = {}

            # Require one two-time matrix per q value.
            if len(two_time_corr_function) != len(q_values):
                raise ValueError(
                    'The number of two-time matrices does not match q.'
                )

            # Iterate over the datasets within the group and the corresponding q values
            for q, t1t2_dataset in zip(q_values, two_time_corr_function):
                # Convert each two-time matrix into a pandas DataFrame
                t1t2_df = pd.DataFrame(t1t2_dataset[:])
                # Add the DataFrame to the dictionary with q as the key
                t1t2_dataframes[q] = t1t2_df

        return combined_df, t1t2_dataframes
        
    # Handle exceptions that may occur during data processing
    except (OSError, KeyError, ValueError, IndexError) as e:
        error_message = (
                        "### Error ###:\n"
                        "The file does not belong to 'Sirius version 3' "
                        "or there is an issue with the data.\n"
                        f"{e}"
                    )
        print(error_message)
        return None, None

# Function to process data from sirius_v3 average data
def process_sirius3m_data(file_path, data_type):
    """
    Read averaged Sirius 3 correlation curves.

    Parameters:
        file_path (str): HDF5 file path.
        data_type (str): 'Originals' or 'Rebinneds'.

    Returns:
        pd.DataFrame or None: Columns '# t' and 'g2(q=...)'.
            None: read failure or invalid data_type.

    Notes:
        Originals: 'Average One-TCF'.
        Rebinneds: times from 'Multi-tau'; curves from 'Average Multi-tau'.
        q values: 'Multi-tau Fitting results'.
    """
    try:
        # Validate the selected data representation before reading.
        if data_type not in ('Originals', 'Rebinneds'):
            raise ValueError(
                "data_type must be 'Originals' or 'Rebinneds'."
            )
        
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
    except (OSError, KeyError, ValueError, IndexError) as e:
        error_message = (
                        "### Error ###:\n"
                        "The file does not belong to 'Sirius 3 average data' "
                        "or there is an issue with the data.\n"
                        f"{e}"
                    )
        print(error_message)
        return None
 
# Function to process data from sirius_v3
def process_sirius3_fit(file_path,data_type):
    """
    Read stored Sirius 3 KWW fit parameters.

    Parameters:
        file_path (str): HDF5 file path.
        data_type (str): 'Originals' or 'Rebinneds'.

    Returns:
        pd.DataFrame or None: Stored fit parameters.
            Columns: 'q_values', 'Baseline', 'Beta', 'Relax_time', 'Gamma'.
            None: read failure or invalid data_type.

    Notes:
        Originals: 'One-time Fitting results'.
        Rebinneds: 'Multi-tau Fitting results'.
    """
    try:
        # Validate the selected data representation before reading.
        if data_type not in ('Originals', 'Rebinneds'):
            raise ValueError(
                "data_type must be 'Originals' or 'Rebinneds'."
            )
    
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
            
            # Get Baseline
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
    except (OSError, KeyError, ValueError, IndexError) as e:
        error_message = f"### Error ###:\nThere is an issue with the data.\n{e}"
        print(error_message)
        return None
                
# Function to process data from APS
def process_aps_data(file_path):
    """
    Read APS correlation curves.

    Parameters:
        file_path (str): HDF5 file path.

    Returns:
        pd.DataFrame or None: Columns '# t' and 'g2(q=...)'.
            None: read failure or incompatible array shapes.

    Notes:
        Sources: 'exchange/tau', 'xpcs/dqlist', 'exchange/g2avgFIT1'.
        Correlation matrix: rows = delay times; columns = q values.
    """
    try:
        with h5py.File(file_path, 'r') as hdf:
            # Keep the time and q axes as arrays, including singleton axes.
            t_data = np.atleast_1d(
                hdf['exchange']['tau'][:].squeeze()
            )
            q_values = np.atleast_1d(
                hdf['xpcs']['dqlist'][:].squeeze()
            )
            g2_data = hdf['exchange']['g2avgFIT1'][:].squeeze()

            # Require one-dimensional time and q axes.
            if t_data.ndim != 1 or q_values.ndim != 1:
                raise ValueError(
                    'Time and q axes must be one-dimensional.'
                )

            # Restore dimensions removed by squeeze for a single time or q.
            expected_shape = (len(t_data), len(q_values))
            if g2_data.ndim < 2 and 1 in expected_shape:
                g2_data = g2_data.reshape(expected_shape)

            # Require one correlation value per time and q pair.
            if g2_data.shape != expected_shape:
                raise ValueError(
                    'The g2 matrix shape does not match '
                    'the time and q axes.'
                )

            combined_df = pd.DataFrame({'# t': t_data})

            for i, q_value in enumerate(q_values):
                g2_data_q = g2_data[:, i]
                combined_df[f'g2(q={q_value})'] = g2_data_q

            return combined_df

    except (OSError, KeyError, ValueError, IndexError) as e:
        error_message = (
            "### Error ###:\n"
            "The file does not belong to 'APS' "
            "or there is an issue with the data.\n"
            f"{e}"
        )
        print(error_message)
        return None

# Function to process data from ESRF
def process_esrf_data(file_path):
    """
    Read ESRF correlation curves.

    Parameters:
        file_path (str): HDF5 file path.

    Returns:
        pd.DataFrame or None: Columns '# t' and 'g2(q=...)'.
            None: read failure.

    Notes:
        Source: 'entry_0000/correlations/full'.
        Stored correlation matrix: rows = q values; columns = delay times.
        Two-time reading is inactive.
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

            # --- Inactive draft: ---
            # verify HDF5 paths and time axes before enabling.
            
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
    except (OSError, KeyError, ValueError, IndexError) as e:
        error_message = (
                        "### Error ###:\n"
                        "The file does not belong to 'ESRF' "
                        "or there is an issue with the data.\n"
                        f"{e}"
                    )
        print(error_message)
        return None

# Function to process data from ESRF old
def process_esrf_data_old(file_path):
    """
    Read correlation curves from a legacy ESRF layout.

    Parameters:
        file_path (str): HDF5 file path.

    Returns:
        pd.DataFrame or None: Columns '# t' and 'g2(q=...)'.
            None: read failure.

    Notes:
        Source: 'entry_0000/dynamix/correlations'.
        q values: mean of columns 1 and 2 in 'q_index_min_max'.
        Legacy reader; not called by the current notebook workflow.
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

            # Read the first column of the 'timeshift' dataset
            t_data = correlation_group['timeshift'][:, 0]

            # Initialize an empty DataFrame with 't' as the first column
            combined_df = pd.DataFrame({'# t': t_data})

            # Read 'q_index_min_max' and average its second and third columns
            q_index_min_max = correlations_group['q_index_min_max'][:]
            q_values = q_index_min_max[:, 1:3].mean(axis=1)

            # Add 'g2' data for each 'q' to the DataFrame
            for i, q_value in enumerate(q_values):
                g2_data = correlation_group['cf'][:, i]
                combined_df[f'g2(q={q_value})'] = g2_data

        return combined_df
    
    except (OSError, KeyError, ValueError, IndexError) as e:
        error_message = f"### Error ###:\nThere is an issue with the data structure in the file.\n{e}"
        print(error_message)
        return None

# Function to read correlation data and available auxiliary results
def read_hdf5_file(file_path, selected_synchrotron, data_type=None,
                   compare_with_beamline=False):
    """
    Read an HDF5 file using the selected input format.

    Parameters:
        file_path (str): HDF5 file path.
        selected_synchrotron (str): Selected input format.
        data_type (str or None, optional): 'Originals' or 'Rebinneds'.
        compare_with_beamline (bool): Read reference fit parameters.
            Default: False. Supported only for Sirius 3.

    Returns:
        tuple: (dataset_df, t1t2_dfs, fit_parameters_dict).
            - dataset_df (pd.DataFrame or None): Correlation curves.
            - t1t2_dfs (dict): q -> two-time DataFrame.
            - fit_parameters_dict (dict): q -> beamline fit parameters.
            Correlation read failure: (None, {}, {}).

    Notes:
        data_type: used only for Sirius 3 formats.
        Unavailable auxiliary results: empty dictionaries.
        Beamline fit parameters: opt-in, Sirius 3 only.
        Disabled comparison: reference fit parameters are not read.
        Reference errors retain the successfully read correlation curves.
    """
    hdf5_file = os.path.basename(file_path)

    # Initialize auxiliary results for the current file.
    t1t2_dfs = {}
    fit_parameters_dict = {}

    try:
        # Process 'Sirius version 1' data and get the DataFrame.
        if selected_synchrotron == 'Sirius 1':
            dataset_df = process_sirius1_data(file_path)

        # Process 'Sirius version 2' data and get both correlation types.
        elif selected_synchrotron == 'Sirius 2':
            dataset_df, t1t2_dfs = process_sirius2_data(file_path)

        # Process 'Sirius version 3' data and get both correlation types.
        elif selected_synchrotron == 'Sirius 3':
            dataset_df, t1t2_dfs = process_sirius3_data(file_path, data_type)

        # Process 'Sirius 3 average' data and get the DataFrame.
        elif selected_synchrotron == 'Sirius 3 average':
            dataset_df = process_sirius3m_data(file_path, data_type)

        # Process 'APS' data and get the DataFrame.
        elif selected_synchrotron == 'APS':
            dataset_df = process_aps_data(file_path)

        # Process 'ESRF' data and get the DataFrame.
        elif selected_synchrotron == 'ESRF':
            dataset_df = process_esrf_data(file_path)

        # Report a format without an associated reader.
        else:
            raise ValueError(
                f"Unsupported input format: {selected_synchrotron!r}"
            )

        # A failed correlation read prevents analysis of this file.
        if dataset_df is None:
            print(
                f"Warning: Data could not be processed for file "
                f"'{hdf5_file}'. Skipping."
            )
            return None, {}, {}

    except Exception as error:
        print(
            f"Warning: Could not process file '{hdf5_file}'. "
            f"Reason: {error}"
        )
        return None, {}, {}

    # Read reference fit parameters only when the comparison is enabled.
    if compare_with_beamline and selected_synchrotron == 'Sirius 3':
        try:
            fit_results_df = process_sirius3_fit(file_path, data_type)

            if fit_results_df is not None:
                # Keep the parameter mapping; comparison rows are built later.
                fit_parameters_dict, _ = initialize_fit(fit_results_df)
            else:
                print(
                    f"Warning: Beamline fit parameters are unavailable for "
                    f"'{hdf5_file}'. Continuing with the program fits."
                )

        # Reference errors do not discard the correlation curves.
        except Exception as error:
            print(
                f"Warning: Could not prepare beamline fit parameters for "
                f"'{hdf5_file}'. Reason: {error}. "
                f"Continuing with the program fits."
            )

    return dataset_df, t1t2_dfs, fit_parameters_dict

#-----------------------------------------------------------------------#
#--------------------- Data Cleaning and Selection ---------------------#
#-----------------------------------------------------------------------#

# Function to extract q values from correlation column names
def extract_q_values(dataset_df):
    """
    Extract q values from correlation column names.

    Parameters:
        dataset_df (pd.DataFrame): Data containing 'g2(q=...)' columns.

    Returns:
        dict: Column name -> q value (float).
            Empty if no valid correlation columns are found.

    Notes:
        Integer, decimal and scientific notation.
        Invalid or nonfinite q values: ignored.
        Original column names: preserved.
    """
    q_values_by_column = {}

    for column in dataset_df.columns:
        # Identify correlation columns before converting the numeric value.
        match = re.fullmatch(r"g2\(q=(.+)\)", str(column))
        if match is None:
            continue

        # float() supports decimal and scientific notation.
        try:
            q_value = float(match.group(1))
        except ValueError:
            continue

        # Store only finite q values.
        if np.isfinite(q_value):
            q_values_by_column[column] = q_value

    return q_values_by_column

# Function to clean the arrays
def clean_series(t, g2, min_points=8, return_pandas=True):
    """
    Remove nonfinite time-correlation pairs.

    Parameters:
        t (array-like): One-dimensional delay times.
        g2 (array-like): Correlation values paired with t.
        min_points (int): Minimum remaining samples. Default: 8.
        return_pandas (bool): Return Series. Default: True.

    Returns:
        tuple: (t_clean, g2_clean).
            - t_clean (pd.Series, np.ndarray or None): Cleaned times.
            - g2_clean (pd.Series, np.ndarray or None): Cleaned correlations.
            Too few samples: (None, None).

    Raises:
        ValueError: Invalid numeric values, dimensions or lengths.
        TypeError: Inputs cannot be converted to numeric arrays.

    Notes:
        Leading zero time: removed after filtering nonfinite pairs.
        Sample order: preserved.
        Series outputs: new integer indices starting at zero.
    """
    # Convert inputs to numeric arrays for positional processing.
    t_arr  = np.asarray(t, dtype=float)
    g2_arr = np.asarray(g2, dtype=float)

    # Require one correlation value per delay time.
    if t_arr.ndim != 1 or g2_arr.ndim != 1:
        raise ValueError('t and g2 must be one-dimensional.')
    if t_arr.shape != g2_arr.shape:
        raise ValueError('t and g2 must have the same length.')

    # Remove both members of each pair containing a nonfinite value.
    mask = np.isfinite(t_arr) & np.isfinite(g2_arr)
    t_arr, g2_arr = t_arr[mask], g2_arr[mask]

    # Remove a zero-delay sample only at the start of the filtered series.
    if t_arr.size > 0 and t_arr[0] == 0.0:
        t_arr, g2_arr = t_arr[1:], g2_arr[1:]

    # Skip curves with too few remaining samples.
    if t_arr.size < min_points:
        return None, None

    if return_pandas:
        return pd.Series(t_arr), pd.Series(g2_arr)
    return t_arr, g2_arr

# Function to get de t range
def get_t_range_slider(t_values):
    """
    Select a delay-time range using Tkinter sliders.

    Parameters:
        t_values (array-like): One-dimensional delay times.

    Returns:
        tuple: (lower_limit, upper_limit).
            - lower_limit (int): First included row position.
            - upper_limit (int): Last included row position.

    Notes:
        Limits are positions, not time values or pandas index labels.
        Closing the window: full range.
        Empty input: (0, -1).
        Left/Right: decrease/increase the lower position by one.
        Down/Up: decrease/increase the upper position by one.
        Arrow controls work throughout the window without duplicate steps.
        Modified shortcuts retain their widget-specific behavior.
    """
    t_values = list(t_values)

    if len(t_values) == 0:
        return 0, -1

    lower_limit = 0
    upper_limit = len(t_values) - 1

    def center_window(window, width=400, height=230):
        """Center a Tkinter window on the screen."""
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x_position = (screen_width - width) // 2
        y_position = (screen_height - height) // 2
        window.geometry(f"{width}x{height}+{x_position}+{y_position}")

    def update_lower_label(value):
        """Update the lower-limit label."""
        index = int(float(value))
        lower_slider_label.config(text=f"{t_values[index]:.5g}")

    def update_upper_label(value):
        """Update the upper-limit label."""
        index = int(float(value))
        upper_slider_label.config(text=f"{t_values[index]:.5g}")

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
        """Move one position and stop the native widget arrow action."""
        if event.keysym == "Left":
            lower_slider.set(max(0, lower_slider.get() - 1))
        elif event.keysym == "Right":
            lower_slider.set(min(len(t_values) - 1, lower_slider.get() + 1))
        elif event.keysym == "Up":
            upper_slider.set(min(len(t_values) - 1, upper_slider.get() + 1))
        elif event.keysym == "Down":
            upper_slider.set(max(0, upper_slider.get() - 1))

        return "break"

    def bind_arrow_keys(widget):
        """Handle range arrows before each widget's native bindings."""
        widget.bindtags((arrow_tag,) + widget.bindtags())
        for child in widget.winfo_children():
            bind_arrow_keys(child)

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
    lower_slider.pack()

    lower_slider_label = tk.Label(
        root, text=f"{t_values[lower_limit]:.5g}", width=12
    )
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
    upper_slider.pack()

    upper_slider_label = tk.Label(
        root, text=f"{t_values[upper_limit]:.5g}", width=12
    )
    upper_slider_label.pack()

    tk.Label(
        root,
        text="Left/Right: lower limit | Down/Up: upper limit"
    ).pack(pady=(4, 0))

    apply_button = tk.Button(root, text="Apply Range", command=apply_range)
    apply_button.pack(pady=8)

    # A leading bind tag prevents Scale bindings from moving a second time.
    arrow_tag = "XPCSTimeRangeKeys"
    for key in ("Left", "Right", "Up", "Down"):
        root.bind_class(arrow_tag, f"<{key}>", on_arrow_key_press)
        for modifier in ("Shift", "Control", "Alt", "Meta"):
            root.bind_class(
                arrow_tag, f"<{modifier}-{key}>", lambda event: None
            )
    bind_arrow_keys(root)

    root.protocol("WM_DELETE_WINDOW", cancel_range)

    # Initialize slider values after both labels exist.
    lower_slider.set(lower_limit)
    upper_slider.set(upper_limit)

    # Bring the window to the front.
    root.lift()
    root.attributes("-topmost", True)
    root.after(200, lambda: root.attributes("-topmost", False))
    root.after_idle(root.focus_force)

    root.mainloop()

    return lower_limit, upper_limit

# Function to get the t range with graphical preview
def get_t_range_slider_preview(dataset_df, preview_q_column=None):
    """
    Select a delay-time range with a live curve preview.

    Parameters:
        dataset_df (pd.DataFrame): '# t' and g2(q=...) columns.
        preview_q_column (str or None): Initial preview column.

    Returns:
        tuple: (lower_limit, upper_limit).
            - lower_limit (int): First included row position.
            - upper_limit (int): Last included row position.

    Raises:
        ValueError: Missing '# t' or correlation columns.

    Notes:
        Default preview: first correlation column.
        Plot: finite pairs with positive delay times; logarithmic x-axis.
        Preview filtering preserves the original selection positions.
        No fitting during slider movement.
        Closing the window or Use Full Range: full range.
        Empty time axis: (0, -1).
        Left/Right: decrease/increase the lower position by one.
        Down/Up: decrease/increase the upper position by one.
        Arrow controls work on the plot, sliders and closed q selector.
        The open q list retains its native keyboard navigation.
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
        return 0, -1

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
        """Return positive finite span limits, or None for an empty span."""
        # Restrict display limits to the selected original row positions.
        selected_t = t_values[lo_index:hi_index + 1]
        positive_t = selected_t[
            np.isfinite(selected_t) & (selected_t > 0)
        ]

        if positive_t.size == 0:
            return None

        return positive_t.min(), positive_t.max()

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
            span_limits = get_safe_span_limits(lo, hi)
            if span_limits is not None:
                ax.axvspan(*span_limits, alpha=0.25)

        ax.set_xlabel("Delay time (s)")
        ax.set_ylabel(r"$g_2(q,t)$")
        ax.set_title(f"Range preview: {q_col}")
        ax.grid(True, alpha=0.3)

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

    def on_arrow_key_press(event):
        """Move one position and stop the native widget arrow action."""
        if event.keysym == "Left":
            lower_slider.set(max(0, lower_slider.get() - 1))
        elif event.keysym == "Right":
            lower_slider.set(min(len(t_values) - 1, lower_slider.get() + 1))
        elif event.keysym == "Up":
            upper_slider.set(min(len(t_values) - 1, upper_slider.get() + 1))
        elif event.keysym == "Down":
            upper_slider.set(max(0, upper_slider.get() - 1))

        return "break"

    def bind_arrow_keys(widget):
        """Handle range arrows before each widget's native bindings."""
        widget.bindtags((arrow_tag,) + widget.bindtags())
        for child in widget.winfo_children():
            bind_arrow_keys(child)

    root = tk.Tk()
    root.title("Select 't' Range")
    root.resizable(True, True)
    center_window(root)
    root.minsize(600, 520)

    # Allocate controls first so a resized plot cannot clip the buttons.
    controls_frame = ttk.Frame(root)
    controls_frame.pack(side='bottom', fill='x')

    # Main plot
    figure = Figure(figsize=(7.2, 3.6), layout='tight')
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
    selector_frame = ttk.Frame(controls_frame)
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
        controls_frame,
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
        controls_frame,
        text=f"{t_values[lower_limit]:.5g}",
        width=14
    )
    lower_slider_label.pack()

    # Upper-limit slider
    upper_slider = tk.Scale(
        controls_frame,
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
        controls_frame,
        text=f"{t_values[upper_limit]:.5g}",
        width=14
    )
    upper_slider_label.pack()

    range_label = tk.Label(controls_frame, text="")
    range_label.pack(pady=4)

    tk.Label(
        controls_frame,
        text="Left/Right: lower limit | Down/Up: upper limit"
    ).pack()

    # Buttons
    button_frame = ttk.Frame(controls_frame)
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

    # Bind before widget classes; the lazily created q popup stays native.
    arrow_tag = "XPCSTimeRangeKeys"
    for key in ("Left", "Right", "Up", "Down"):
        root.bind_class(arrow_tag, f"<{key}>", on_arrow_key_press)
        for modifier in ("Shift", "Control", "Alt", "Meta"):
            root.bind_class(
                arrow_tag, f"<{modifier}-{key}>", lambda event: None
            )
    bind_arrow_keys(root)

    root.protocol("WM_DELETE_WINDOW", use_full_range)

    # Initialize slider values after all labels/widgets exist.
    lower_slider.set(lower_limit)
    upper_slider.set(upper_limit)
    update_range_labels()

    root.lift()
    root.attributes("-topmost", True)
    root.after(200, lambda: root.attributes("-topmost", False))
    root.after_idle(root.focus_force)

    root.mainloop()

    return lower_limit, upper_limit

# Function to obtain 't' range limits
def get_t_range_limits(dataset_df, define_t_range, show_preview=False):
    """
    Obtain delay-time selection positions.

    Parameters:
        dataset_df (pd.DataFrame): Data containing '# t'.
        define_t_range (bool): Enable interactive range selection.
        show_preview (bool): Use the plot preview. Default: False.

    Returns:
        tuple: (lower_limit, upper_limit).
            - lower_limit (int): First included row position.
            - upper_limit (int): Last included row position.

    Raises:
        ValueError: Missing '# t'; missing correlation columns for preview.

    Notes:
        define_t_range=False: full range.
        Both limits are inclusive.
        Empty time axis: (0, -1).
        The main program enables preview for single-file processing.
    """
    if '# t' not in dataset_df.columns:
        raise ValueError("The DataFrame must contain a '# t' column.")

    # An empty axis has no selectable row positions.
    if len(dataset_df['# t']) == 0:
        return 0, -1
    
    if not define_t_range:
        return 0, len(dataset_df['# t']) - 1

    if show_preview:
        return get_t_range_slider_preview(dataset_df)

    t_values = dataset_df['# t'].tolist()
    return get_t_range_slider(t_values)

# Function to obtain the time bounds of a selected interval
def get_time_bounds(t):
    """
    Obtain the finite delay-time bounds of a selected interval.

    Parameters:
        t (array-like): Selected delay times.

    Returns:
        tuple or None: (t_min, t_max).
            - t_min (float): Minimum selected time.
            - t_max (float): Maximum selected time.
            No finite times: None.
    """
    # Ignore nonfinite times when obtaining the interval bounds.
    t_values = np.asarray(t, dtype=float)
    finite_times = t_values[np.isfinite(t_values)]

    if finite_times.size == 0:
        print("Warning: The selected interval contains no finite delay times.")
        return None

    return float(finite_times.min()), float(finite_times.max())

# Function to prompt the user for defining the q values
def ask_user_for_select_q():
    """
    Ask whether to select q curves interactively.

    Returns:
        bool: True for Yes; False for No or a closed dialog.

    Notes:
        Curve selection: performed later by GraphWindow.
    """
    root = tk.Tk()
    root.withdraw()

    # Destroy the hidden window after the dialog, including on errors.
    try:
        user_response = messagebox.askyesno(
            "Select q Values",
            "Do you want to select q values?",
            parent=root
        )
    finally:
        root.destroy()

    return user_response

#-----------------------------------------------------------------------#
#----------------------------- Data Export -----------------------------#
#-----------------------------------------------------------------------#

# Function to save DataFrame to a TSV file with the appropriate line terminator argument
def save_dataframe_to_tsv(dataset_df, output_dataframe, header=True):
    """
    Write a DataFrame as tab-separated text.

    Parameters:
        dataset_df (pd.DataFrame): Data to export.
        output_dataframe (str, os.PathLike or io.TextIOBase):
            Output path or open text file.
        header (bool): Write column names. Default: True.

    Returns:
        None: Data written to the destination.

    Raises:
        OSError: Destination cannot be written.
        TypeError: Export error unrelated to the line terminator argument.

    Notes:
        Tab separator; no DataFrame index; LF line endings.
        Legacy pandas: retry with 'line_terminator'.
        Output paths: UTF-8 encoding; existing files overwritten.
        Open text files must use newline='' to preserve line endings.
    """
    try:
        # Try using 'lineterminator'
        dataset_df.to_csv(
            output_dataframe,
            sep='\t',
            index=False,
            lineterminator='\n',
            header=header,
            encoding='utf-8'
        )
    except TypeError as error:
        # Retry only when the installed pandas rejects this argument.
        if "unexpected keyword argument 'lineterminator'" not in str(error):
            raise

        # If that fails, use 'line_terminator'
        dataset_df.to_csv(
            output_dataframe,
            sep='\t',
            index=False,
            line_terminator='\n',
            header=header,
            encoding='utf-8'
        )

# Function to save t1-vs-t2 data to a TSV file
def save_t1t2_data(t1t2_dfs, directory, base_name):
    """
    Export two-time correlation matrices to individual .dat files.

    Parameters:
        t1t2_dfs (dict): q values -> pd.DataFrame matrices.
        directory (str): Parent output directory.
        base_name (str): Output filename prefix.

    Returns:
        None: Files written to the 't1-vs-t2' directory.

    Raises:
        AnalysisStopped: Directory creation or file writing fails.

    Notes:
        First line: '# q value: ...'.
        Matrix rows: tab-separated; no index or column names.
        Suffixes: a, ..., z, aa, ab, ...; dictionary iteration order.
        Matrix row and column order: preserved.
        Existing output files are overwritten.
        A failed export may leave an incomplete output file.
    """
    # Define the new 't1-vs-t2' directory.
    t1t2_directory = os.path.join(directory, 't1-vs-t2')

    try:
        # Create the output directory if it does not exist.
        os.makedirs(t1t2_directory, exist_ok=True)

        for i, (q, t1t2_df) in enumerate(t1t2_dfs.items(), start=1):
            # Get the alphabetic suffix corresponding to the index.
            # Continue with aa, ab, ... after z.
            letter = generate_alphabetic_suffix(i)

            # Create the file name with the suffix.
            t1t2_file = os.path.join(
                t1t2_directory, f"{base_name}_t1-t2_{letter}.dat")

            # Save q as a comment, followed by the matrix values.
            with open(t1t2_file, 'w', encoding='utf-8', newline='') as f:
                f.write(f"# q value: {q}\n")
                save_dataframe_to_tsv(t1t2_df, f, header=False)

    except OSError as error:
        if error.errno == errno.ENOSPC:
            explanation = (
                "There is not enough free space on the destination filesystem.\n"
                "Free space or choose another output directory, then run again.")
        else:
            explanation = "Check the output directory and its write permissions."

        # Stop with a readable message, including the technical details.
        raise AnalysisStopped(
            "Analysis stopped while exporting two-time correlations.\n"
            f"Output directory: {t1t2_directory}\n"
            f"{explanation}\n"
            f"Technical details: {error}"
        ) from None

#-----------------------------------------------------------------------#
#--------------------------- Initializations ---------------------------#
#-----------------------------------------------------------------------#

# Function to initialize error and success counters
def initialize_error_and_success_counters():
    """
    Initialize processing counters and record lists.

    Returns:
        dict: Counters and processing records.
            - Counter values (int): Initialized to 0.
            - Record lists (list): Initialized empty.

    Notes:
        q_values: Recognized input correlation columns.
        excluded_q: Curves excluded by manual q selection.
        skipped_q: Curves skipped before fitting.
        success: Curves with three finite fit R^2 values.
        failure: Curves with at least one nonfinite fit R^2.
        below_threshold: Individual model fits at or below the threshold.
        Counter updates occur outside this function.
    """
    counters = {
        'hdf5_files': 0,
        'q_values': 0,
        'excluded_q': 0,
        'skipped_q': 0,
        'invalid_data': 0,
        'success': 0,
        'failure': 0,
        'below_threshold': 0,
        'invalid_files': [],
        'failed_files': [],
        'below_threshold_files': []
    }

    return counters

# Function to initialize parameter tracking for averages
def initialize_data_for_parameter_averages():
    """
    Initialize parameter accumulators for each fitting model.

    Returns:
        dict: Nested dictionaries for parameter tracking.
            - 'single' (dict): Converted C/q^2 quotient.
            - 'stretched' (dict): Converted C/q^2 quotient and gamma.
            - 'cumulants' (dict): Converted C1/q^2 quotient and PDI.

    Notes:
        'fit_count': Number of included fits.
        '* sum': Parameter sums, initialized to 0.
        '* values': Individual parameter values, initialized empty.
        Fit selection occurs outside this function.
    """
    av_params = {
        'single': {
            'fit_count': 0,
            'Diff_coef sum': 0,
            'Diff_coef values': []
        },
        'stretched': {
            'fit_count': 0,
            'Diff_coef sum': 0,
            'Diff_coef values': [],
            'Gamma sum': 0,
            'Gamma values': []
        },
        'cumulants': {
            'fit_count': 0,
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
    Map reference fit parameters by q value.

    Parameters:
        fit_results_df (pd.DataFrame): Reference fit results.
            Required numeric columns:
            'q_values', 'Baseline', 'Beta', 'Relax_time', 'Gamma'.

    Returns:
        tuple: (fit_parameters_dict, fit_variations).
            - fit_parameters_dict (dict): q -> parameter dictionary.
            - fit_variations (list): Empty list for percentage differences.

    Raises:
        TypeError: Input is not a pandas DataFrame.
        ValueError: Required columns are missing.

    Notes:
        Parameter keys: 'Baseline', 'Beta', 'Relax_time', 'Gamma'.
        q values retain their stored precision.
        Duplicate q values: the last row replaces earlier rows.
        Empty input with the required columns returns ({}, []).
    """
    # Require a DataFrame with the reference fit parameters.
    if not isinstance(fit_results_df, pd.DataFrame):
        raise TypeError("fit_results_df must be a pandas DataFrame.")

    # Check the columns required by the parameter mapping.
    required_columns = [
        'q_values', 'Baseline', 'Beta', 'Relax_time', 'Gamma'
    ]
    missing_columns = [
        column for column in required_columns
        if column not in fit_results_df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing fit-result columns: {missing_columns}"
        )

    # Initialize an empty dictionary to store the mapping of q_values to fit parameters
    fit_parameters_dict = {}
    
    # Iterate over each row in the HDF5 fit results DataFrame
    for _, row in fit_results_df.iterrows():
        # Preserve the q value stored in the HDF5 fit results.
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
    Create two figures, plotting axes and a discrete colormap.

    Parameters:
        q_len (int): Number of q curves. Must be at least 1.

    Returns:
        tuple: (fig1, axes1, fig2, axes2, cmap, lines_labels_dict).
            - fig1 (matplotlib.figure.Figure): Correlation fits and
                single exponential results.
            - axes1 (tuple): Five matplotlib.axes.Axes objects.
            - fig2 (matplotlib.figure.Figure): Generalized exponential
                and cumulants results.
            - axes2 (tuple): Six matplotlib.axes.Axes objects.
            - cmap (matplotlib.colors.Colormap): q_len color entries.
            - lines_labels_dict (dict): Empty legend-tracking lists.

    Raises:
        ValueError: q_len is less than 1.

    Notes:
        Each figure contains a 2-row, 3-column grid.
        axes1: (ax_single, ax_stretched, ax_cumulants,
                ax_Cq2_si, ax_Cq_si).
        axes2: (ax_Cq2_st, ax_Cq_st, ax_gammaq,
                ax_Cq2_cu, ax_Cq_cu, ax_PDIq).
        The sixth axes on fig1 is hidden and excluded from axes1.
        Figure display and export occur outside this function.
    """
    # Require at least one curve before creating the figures.
    if q_len < 1:
        raise ValueError("q_len must be at least 1.")

    # Create two figures for each HDF5 file, each with a 2x3 subplot grid.
    fig1 = plt.figure(figsize=(20, 16))
    axs1 = fig1.subplots(2, 3)

    fig2 = plt.figure(figsize=(20, 16))
    axs2 = fig2.subplots(2, 3)

    # Optional full-figure titles
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

    # Explain the displayed quantities in the unused subplot.
    ax_si.text(0.02, 0.98, "Plot labels",
               transform=ax_si.transAxes, fontsize=14, fontweight='bold',
               ha='left', va='top')
    ax_si.text(0.02, 0.88,
               "Baseline: fitted correlation offset.\n"
               r"$\beta$: fitted correlation amplitude." "\n"
               "Gamma: temporal relaxation exponent.\n"
               r"PDI: relative rate variance, $C_2/C_1^2$." "\n"
               r"$C$: inverse time scale; $C_1$ for cumulants." "\n\n"
               "Mean/Av.: average over selected q values.\n"
               r"$C/q^2$: rate-to-$q^2$ quotient." "\n"
               r"$D_{\mathrm{app}}$: slope of $C$ versus $q^2$." "\n"
               r"$K$: prefactor in $C = Kq^n$." "\n"
               "Diffusion interpretation requires a suitable model.\n"
               "SD: standard deviation across selected q values.\n"
               "Parentheses in means: SD.\n"
               "Parentheses in slope: fit standard error.\n\n"
               "Exponent n: dependence of C on q.\n"
               r"$R^2$: coefficient of determination." "\n"
               "Pearson r: linear correlation coefficient.\n"
               "Units assume t in s and q in angstrom^-1.",
               transform=ax_si.transAxes, fontsize=11,
               ha='left', va='top', linespacing=1.4)

    # Access individual subplots on the second page
    ax_Cq2_st = axs2[0, 0]
    ax_Cq_st = axs2[0, 1]
    ax_gammaq = axs2[0, 2]
    ax_Cq2_cu = axs2[1, 0]
    ax_Cq_cu = axs2[1, 1]
    ax_PDIq = axs2[1, 2]

    # Make the axes boxes square without forcing equal data scales.
    for ax in [ax_single, ax_stretched, ax_cumulants, ax_Cq2_si, ax_Cq_si, ax_Cq2_st, ax_Cq_st, ax_gammaq, ax_Cq2_cu, ax_Cq_cu, ax_PDIq]:
        ax.set_box_aspect(1)

    # Adjust subplot parameters to reduce margins
    fig1.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.25, hspace=0.25)
    fig2.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.25, hspace=0.25)

    # Initialize a discrete color map for different q values.
    cmap = plt.get_cmap('gist_rainbow', q_len)

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
    Initialize storage for derived parameters and selected fit results.

    Returns:
        tuple: (derived_params, qualified_params).
            - derived_params (dict): Empty lists for q, q squared,
                colors, relaxation rates, gamma and PDI.    
            - qualified_params (dict): Empty lists for selected q values,
                relaxation rates, colors, gamma and PDI.

    Notes:
        Fit selection occurs outside this function using R^2 > threshold.
        Selected q values and colors are stored separately for each model.
        Selected gamma values correspond to the generalized exponential.
        Selected PDI values correspond to the cumulants model.
    """
    # Initialize empty lists for derived parameters
    derived_params = {
        "q_value": [],
        "q_square": [],
        "color": [],   
        "relax_rate_single": [],
        "relax_rate_stretched": [],
        "gamma": [],  
        "relax_rate_cumulants": [],
        "PDI": []
    }

    # Initialize empty lists for qualified parameters
    # Keep separate q and color lists for each model.
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
    Display correlation curves and select q columns interactively.

    Attributes:
        root (tk.Tk): Selection window.
        new_dataset_df (pd.DataFrame): Input or selected correlation data.
        lower_limit (int): First original row position for the preview.
        upper_limit (int): Last original row position for the preview.
        cmap (matplotlib.colors.Colormap): Curve colors.
        figure (matplotlib.figure.Figure): Preview figure.
        ax (matplotlib.axes.Axes): Preview axes.
        canvas (FigureCanvasTkAgg): Matplotlib canvas embedded in Tkinter.
        selected_values (list): Selected q values.
        selected_columns (list): Selected DataFrame column names.
        lines_labels_dict (dict): Curve objects and display labels.
        q_buttons (list): Checkbox widgets and BooleanVar objects.
        plotted_columns (list): q columns paired with preview line objects.

    Notes:
        Time limits are inclusive row positions, not delay-time values.
        Preview curves use the single exponential model.
        Selecting q columns preserves the DataFrame rows.
    """
    def __init__(self, root, new_dataset_df, lower_limit, upper_limit, cmap):
        """
        Initialize the selection window, preview and controls.

        Parameters:
            root (tk.Tk): Existing selection window.
            new_dataset_df (pd.DataFrame): '# t' and g2 columns.
            lower_limit (int): First preview row position.
            upper_limit (int): Last preview row position.
            cmap (matplotlib.colors.Colormap): Curve colors.

        Returns:
            None: Window contents and instance attributes initialized.

        Notes:
            The caller starts the event loop and destroys the window.
        """
        # Initialize the root window
        self.root = root
        self.root.title("Select q values")

        # Treat the X button as Exit
        self.root.protocol("WM_DELETE_WINDOW", self.exit_window)

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
        """
        Create normalized preview curves for q selection.

        Returns:
            None: Figure, canvas and curve records initialized.

        Notes:
            Clean each preview series before fitting; require eight points.
            Unavailable previews retain their q selection checkbox.
            Input DataFrame values and rows are preserved.
        """
        self.figure = Figure(figsize=(8, 5))
        self.ax = self.figure.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=1)

        # Track one line pair per selectable q column.
        self.plotted_columns = []

        for i, q_column in enumerate(self.new_dataset_df.columns[1:], 1):

            # Read the complete q value, including scientific notation.
            match = re.search(
                r"q=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\)",
                q_column)

            if not match:
                continue

            q_value = float(match.group(1))

            t_q = self.new_dataset_df['# t']
            g2_q = self.new_dataset_df[q_column]

            # Apply the inclusive original row positions before cleaning.
            t_q = t_q.iloc[self.lower_limit:self.upper_limit + 1]
            g2_q = g2_q.iloc[self.lower_limit:self.upper_limit + 1]

            # Use the same cleaning and minimum sample count as the main fit.
            t_q, g2_q = clean_series(
                t_q, g2_q, min_points=8, return_pandas=False)

            # Empty lines retain a checkbox when no preview is available.
            preview_t = np.array([])
            normalized_g2 = np.array([])
            normalized_fit = np.array([])
            fit_label = "Fewer than 8 valid points"

            if t_q is not None:
                fit_params_selq, fitted_curve_selq, r2_selq = (
                    fit_single_exponential(t_q, g2_q))
                A_selq, B_selq = fit_params_selq[:2]
                fit_label = f"R^2: {r2_selq:.3f}"

                # Normalize only with finite parameters and nonzero amplitude.
                if np.isfinite(A_selq) and np.isfinite(B_selq) and B_selq != 0:
                    # The logarithmic time axis displays positive times only.
                    positive_time = t_q > 0
                    preview_t = t_q[positive_time]

                    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                        normalized_g2 = (g2_q[positive_time] - A_selq) / B_selq
                        normalized_fit = (fitted_curve_selq[positive_time] - A_selq) / B_selq

                    # Keep gaps for undefined normalized values.
                    normalized_g2 = np.where(
                        np.isfinite(normalized_g2), normalized_g2, np.nan)
                    normalized_fit = np.where(
                        np.isfinite(normalized_fit), normalized_fit, np.nan)
                else:
                    fit_label = "Preview unavailable: invalid normalization"

            color = self.cmap(i - 1)

            line_exp_sq = self.ax.semilogx(
                preview_t, normalized_g2, 'o', color=color)[0]
            line_fit_sq = self.ax.semilogx(
                preview_t, normalized_fit, color=color, linestyle='-')[0]

            self.plotted_columns.append(q_column)
            self.lines_labels_dict["exp_lines_selq"].append(line_exp_sq)
            self.lines_labels_dict["exp_labels_selq"].append(f"q = {q_value:.6g}")
            self.lines_labels_dict["fit_lines_selq"].append(line_fit_sq)
            self.lines_labels_dict["fit_labels_selq"].append(fit_label)

        self.ax.set_title('Correlation function fitting - Single Exponential Model')
        self.ax.set_xlabel('Delay Time (s)')
        self.ax.set_ylabel(r'$(g_2 - \mathrm{baseline}) / \beta$')

        # Do not add a legend inside the plot.
        # The q and R^2 labels are shown in the checkbox list below the graph.
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
            zip(self.plotted_columns,
                self.lines_labels_dict["exp_labels_selq"],
                self.lines_labels_dict["fit_labels_selq"],
                self.lines_labels_dict["exp_lines_selq"])):

            # Read the complete q value, including scientific notation.
            match = re.search(
                r"q=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\)",
                q_column)

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
            self.plotted_columns,
            self.lines_labels_dict["exp_lines_selq"]
        ):
            if line.get_visible():
                # Read the complete q value, including scientific notation.
                match = re.search(
                    r"q=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\)",
                    column)

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
        """
        Apply the selected q columns and stop the event loop.

        Returns:
            pd.DataFrame or None: Filtered data; None if no q was selected.
        """
        # Keep the window open when no curve is selected.
        if not self.selected_columns:
            messagebox.showwarning(
                "No q values selected",
                "Select at least one q value, or use Exit to keep all curves.",
                parent=self.root
            )
            return

        #print("Selected values:", self.selected_values)

        # Filter the input columns using the selected q columns
        selected_columns = ['# t'] + self.selected_columns
        filtered_dataset_df = self.new_dataset_df[selected_columns]

        # Store the filtered DataFrame
        self.new_dataset_df = filtered_dataset_df

        # Stop the event loop; the caller destroys the window
        self.root.quit()

        # Return the filtered DataFrame
        return filtered_dataset_df

    def exit_window(self):
        """
        Stop the selection without applying checkbox changes.

        Returns:
            None

        Notes:
            Input DataFrame: unchanged.
            The caller destroys the window after mainloop().
        """
        self.root.quit()

# Function to select q columns in an interactive window
def select_q_columns(dataset_df, lower_limit, upper_limit, cmap):
    """
    Select q curves in an interactive window.

    Parameters:
        dataset_df (pd.DataFrame): Input correlation data.
        lower_limit (int): First preview row position.
        upper_limit (int): Last preview row position, inclusive.
        cmap (matplotlib.colors.Colormap): Curve colors.

    Returns:
        pd.DataFrame: Input data or selected q columns.

    Notes:
        Select: apply the selected q columns.
        Exit or X: keep the input data.
        Time limits affect the preview only. DataFrame rows are preserved.
    """
    # Create the Tkinter window.
    root = tk.Tk()

    try:
        # Create an instance of the GraphWindow class.
        graph_window = GraphWindow(
            root, dataset_df, lower_limit, upper_limit, cmap)

        # Start the Tkinter event loop.
        root.mainloop()

        # Get the stored result without applying the selection again.
        return graph_window.new_dataset_df
    finally:
        # Close the Tkinter window after the event loop or an error.
        root.destroy()

#-----------------------------------------------------------------------#
#----------------------------- Fit Models ------------------------------#
#-----------------------------------------------------------------------#

# Optional function to fit a model without parameter bounds
def fit_model(t, g2, model_func, initial_params):
    """
    Fit a model without parameter bounds.

    Parameters:
        t (array-like): One-dimensional delay times.
        g2 (array-like): One-dimensional correlation values.
        model_func (callable): Model with signature model_func(t, *params).
        initial_params (array-like): Initial parameter estimates.

    Returns:
        tuple: (fit_params, fitted_curve, r2).
            - fit_params (numpy.ndarray): Optimized parameters.
            - fitted_curve (numpy.ndarray): Model values at t.
            - r2 (float): Coefficient of determination.
            Numerical failures return NaN parameters, curve and score.

    Notes:
        Matching lengths; finite inputs; nonconstant g2.
        At least max(2, number of parameters) observations.
        Data cleaning occurs before fitting.
        Optimizer warnings are suppressed; covariance is not returned.
        Optional alternative; not called by the main workflow.
    """
    n_points = len(t)
    n_params = len(initial_params)

    try:
        # Convert inputs to one-dimensional floating-point arrays.
        t = np.asarray(t, dtype=float)
        g2 = np.asarray(g2, dtype=float)
        initial_params = np.asarray(initial_params, dtype=float)

        # Require matching series and enough observations for the fit and R^2.
        if (
            t.ndim != 1
            or g2.ndim != 1
            or initial_params.ndim != 1
            or t.shape != g2.shape
            or n_points < max(2, n_params)
        ):
            raise ValueError(
                "Invalid fit dimensions or insufficient data."
            )

        # Reject nonfinite input values; cleaning occurs before fitting.
        if not (
            np.isfinite(t).all()
            and np.isfinite(g2).all()
            and np.isfinite(initial_params).all()
        ):
            raise ValueError("Fit inputs must contain finite values.")

        # R^2 cannot assess a series with no variation in g2.
        if np.ptp(g2) == 0:
            raise ValueError("R^2 is undefined for a constant series.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore warnings
            fit_params, _ = curve_fit(
                model_func,
                t,
                g2,
                p0=initial_params
            )

        fitted_curve = np.asarray(
            model_func(t, *fit_params),
            dtype=float
        )
        r2 = r2_score(g2, fitted_curve)

        # Keep nonfinite results on the same failure path as optimizer errors.
        if not (
            np.isfinite(fit_params).all()
            and np.isfinite(fitted_curve).all()
            and np.isfinite(r2)
        ):
            raise ValueError("The fit produced nonfinite results.")

        return fit_params, fitted_curve, r2

    except (RuntimeError, ValueError):
        return (
            np.full(n_params, np.nan),
            np.full(n_points, np.nan),
            np.nan
        )
    
# Function to fit a supported model with parameter bounds
def fit_model_with_constraints(t, g2, model_func, initial_params):
    """
    Fit a supported model with parameter bounds.

    Parameters:
        t (array-like): One-dimensional delay times.
        g2 (array-like): One-dimensional correlation values.
        model_func (callable): Supported model function.
        initial_params (array-like): Initial parameter estimates.

    Returns:
        tuple: (fit_params, fitted_curve, r2).
            - fit_params (numpy.ndarray): Optimized parameters.
            - fitted_curve (numpy.ndarray): Model values at t.
            - r2 (float): Coefficient of determination.
            Numerical failures return NaN parameters, curve and score.

    Raises:
        ValueError: Unsupported model name.

    Notes:
        Bounds are selected using the model function name.
        Matching lengths; finite inputs; nonconstant g2.
        At least max(2, number of parameters) observations.
        Zero lower bounds permit zero.
        Optimizer warnings are suppressed; covariance is not returned.
    """
    # Parameter bounds for each supported model.
    # Zero lower bounds permit zero; they do not enforce strict positivity.
    model_name = getattr(model_func, '__name__', None)

    if model_name == 'single_exponential':
        # Parameter order: A, B, C. Restriction: A >= 0.
        bounds = (
            [0, -np.inf, -np.inf],
            [np.inf, np.inf, np.inf]
        )

    elif model_name == 'stretched_exponential':
        # Parameter order: A, B, C, gamma.
        # Restrictions: A >= 0, gamma >= 0.
        bounds = (
            [0, -np.inf, -np.inf, 0],
            [np.inf, np.inf, np.inf, np.inf]
        )
        #bounds = ([0, 0, -np.inf, 0], [2, 1, np.inf, 5])

    elif model_name == 'stretched_exponential_refine':
        # Parameter order: C, gamma. A and B are fixed.
        # Restrictions: C >= 0, gamma >= 0.
        bounds = ([0, 0], [np.inf, np.inf])

    elif model_name == 'cumulants_model':
        # Parameter order: A, B, C1, C2.
        # Restrictions: A >= 0, C2 >= 0.
        bounds = (
            [0, -np.inf, -np.inf, 0],
            [np.inf, np.inf, np.inf, np.inf]
        )

    elif model_name == 'cumulants_model_refine':
        # Parameter order: C1, C2. A and B are fixed.
        # Restrictions: C1 >= 0, C2 >= 0.
        bounds = ([0, 0], [np.inf, np.inf])

    else:
        raise ValueError(
            f"Unsupported constrained model: {model_name!r}"
        )

    n_points = len(t)
    n_params = len(initial_params)

    try:
        # Convert inputs to one-dimensional floating-point arrays.
        t = np.asarray(t, dtype=float)
        g2 = np.asarray(g2, dtype=float)
        initial_params = np.asarray(initial_params, dtype=float)

        # Require matching series and enough observations for the fit and R^2.
        if (
            t.ndim != 1
            or g2.ndim != 1
            or initial_params.ndim != 1
            or t.shape != g2.shape
            or n_points < max(2, n_params)
        ):
            raise ValueError(
                "Invalid fit dimensions or insufficient data."
            )

        # Reject nonfinite input values; cleaning occurs before fitting.
        if not (
            np.isfinite(t).all()
            and np.isfinite(g2).all()
            and np.isfinite(initial_params).all()
        ):
            raise ValueError("Fit inputs must contain finite values.")

        # R^2 cannot assess a series with no variation in g2.
        if np.ptp(g2) == 0:
            raise ValueError("R^2 is undefined for a constant series.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore warnings
            fit_params, _ = curve_fit(
                model_func,
                t,
                g2,
                p0=initial_params,
                bounds=bounds
            )

        fitted_curve = np.asarray(
            model_func(t, *fit_params),
            dtype=float
        )
        r2 = r2_score(g2, fitted_curve)

        # Keep nonfinite results on the same failure path as optimizer errors.
        if not (
            np.isfinite(fit_params).all()
            and np.isfinite(fitted_curve).all()
            and np.isfinite(r2)
        ):
            raise ValueError("The fit produced nonfinite results.")

        return fit_params, fitted_curve, r2

    except (RuntimeError, ValueError):
        return (
            np.full(n_params, np.nan),
            np.full(n_points, np.nan),
            np.nan
        )
    
# Single Exponential Model function
def single_exponential(t, A, B, C):
    """
    Evaluate g2(t) = A + B * exp(-2 * C * t).

    Parameters:
        t (float or array-like): Delay times.
        A (float): Baseline.
        B (float): Correlation amplitude.
        C (float): Relaxation rate; inverse time units.

    Returns:
        numpy.float64 or numpy.ndarray: Model values.
    """
    # Support scalar, list, array and pandas Series inputs.
    t = np.asarray(t, dtype=float)

    return A + B * np.exp(-2 * C * t)

# Generalized Exponential Model function
def stretched_exponential(t, A, B, C, gamma):
    """
    Evaluate g2(t) = A + B * exp(-2 * (C * t)**gamma).

    Parameters:
        t (float or array-like): Delay times.
        A (float): Baseline.
        B (float): Correlation amplitude.
        C (float): Relaxation rate; inverse time units.
        gamma (float): Dimensionless relaxation exponent.

    Returns:
        numpy.float64 or numpy.ndarray: Model values.

    Notes:
        0 < gamma < 1: stretched exponential.
        gamma = 1: single exponential.
        gamma > 1: compressed exponential.
        Noninteger gamma requires nonnegative C * t.
    """
    # Support scalar, list, array and pandas Series inputs.
    t = np.asarray(t, dtype=float)

    return A + B * np.exp(-2 * (C * t) ** gamma)

# Cumulants Model function
def cumulants_model(t, A, B, C1, C2):
    """
    Evaluate the second-order cumulants approximation for g2(t).

    Parameters:
        t (float or array-like): Delay times.
        A (float): Baseline.
        B (float): Correlation amplitude.
        C1 (float): First cumulant; inverse time units.
        C2 (float): Second cumulant; inverse time squared units.

    Returns:
        numpy.float64 or numpy.ndarray: Model values.

    Notes:
        g2(t) = A + B * exp(-2 * C1 * t)
                * (1 + 0.5 * C2 * t**2)**2.
    """
    # Support scalar, list, array and pandas Series inputs.
    t = np.asarray(t, dtype=float)

    return A + B * np.exp(-2 * C1 * t) * (1 + (1 / 2) * C2 * t**2)**2

# Function to fit a single exponential model to the provided data
def fit_single_exponential(t, g2):
    """
    Fit A, B and C in the single exponential model.

    Parameters:
        t (array-like): One-dimensional delay times.
        g2 (array-like): One-dimensional correlation values.

    Returns:
        tuple: (fit_params_single, fitted_curve_single, r2_single).
            - fit_params_single (numpy.ndarray): [A, B, C].
            - fitted_curve_single (numpy.ndarray): Model values at t.
            - r2_single (float): Coefficient of determination.
            Numerical failures return NaN parameters, curve and score.

    Notes:
        Initial guesses use the first two and last five observations.
        C0 uses the positive time closest to the estimated midpoint.
        At least three observations; at least one positive time.
        The main workflow applies min_points=8 before fitting.
    """
    n_points = len(t)

    try:
        # Convert inputs to arrays for positional indexing.
        t = np.asarray(t, dtype=float)
        g2 = np.asarray(g2, dtype=float)

        # Validate the series before calculating the initial guesses.
        if (
            t.ndim != 1
            or g2.ndim != 1
            or t.shape != g2.shape
            or n_points < 3
            or not np.isfinite(t).all()
            or not np.isfinite(g2).all()
        ):
            raise ValueError(
                "Invalid single exponential input data."
            )

        # Calculate the initial guess for parameters A and B:
        A0 = np.mean(g2[-5:])     # Average of the last 5 points
        B0 = np.mean(g2[:2]) - 1  # Average of the first 2 points minus 1

        # Calculate the initial guess for parameter C:
        # Obtain y
        y = (np.mean(g2[:2]) + np.mean(g2[-5:])) / 2

        # Find the closest x value in the experimental curve.
        # Use positive times to avoid division by zero or a negative C0.
        positive_indices = np.flatnonzero(t > 0)

        if positive_indices.size == 0:
            raise ValueError(
                "A positive time is required to estimate C0."
            )

        closest_index = positive_indices[
            np.abs(g2[positive_indices] - y).argmin()
        ]
        closest_x = t[closest_index]

        # Obtain C0:
        C0 = 1 / closest_x

        # Initial guess for parameters A, B, and C:
        initial_params_single = [A0, B0, C0]

        # Fit the curve g2 = A + B * exp(-2C * t)

        # With constraints
        fit_params_single, fitted_curve_single, r2_single = (
            fit_model_with_constraints(
                t, g2, single_exponential, initial_params_single)
        )

        # Without constraints
        #fit_params_single, fitted_curve_single, r2_single = (
        #   fit_model(t, g2, single_exponential, initial_params_single)
        #)

        return fit_params_single, fitted_curve_single, r2_single

    except ValueError:
        return (
            np.full(3, np.nan),
            np.full(n_points, np.nan),
            np.nan
        )
    
## Functions to fit a stretched exponential model to the provided data

# Function to fit C and gamma with A and B fixed
def fit_stretched_exponential(t, g2, fit_params_single):
    """
    Fit C and gamma with A and B fixed from the single fit.

    Parameters:
        t (array-like): One-dimensional delay times.
        g2 (array-like): One-dimensional correlation values.
        fit_params_single (array-like): Single-fit parameters [A, B, C].

    Returns:
        tuple: (fit_params_full, fitted_curve, r2).
            - fit_params_full (list): [A, B, C, gamma].
            - fitted_curve (numpy.ndarray): Model values at t.
            - r2 (float): Coefficient of determination.
            Numerical failures return NaN parameters, curve and score.

    Notes:
        A and B remain fixed.
        Initial C: single-fit rate. Initial gamma: 1.
        C and gamma have zero lower bounds.
    """
    # Extract A, B, C from the single_exponential fit.
    fit_params_single = np.asarray(fit_params_single, dtype=float)

    # Stop if the preceding fit did not provide three finite parameters.
    if (
        fit_params_single.shape != (3,)
        or not np.isfinite(fit_params_single).all()
    ):
        return [np.nan] * 4, np.full(len(t), np.nan), np.nan

    A, B, C_single = fit_params_single

    # Initial guesses for C and gamma:
    C0 = C_single
    gamma0 = 1.0
    initial_params = [C0, gamma0]

    # Prepare the model with fixed A and B.
    # Only C and gamma are exposed to the optimizer.
    def stretched_exponential_refine(t, C, gamma):
        return stretched_exponential(t, A, B, C, gamma)

    # Fit with constraints
    fit_params, fitted_curve, r2 = fit_model_with_constraints(
        t, g2, stretched_exponential_refine, initial_params
    )

    # fit_params currently contains [C, gamma]
    # We want [A, B, C, gamma]
    if np.isfinite(fit_params).all():
        C_fit, gamma_fit = fit_params
        fit_params_full = [A, B, C_fit, gamma_fit]
    else:
        fit_params_full = [np.nan, np.nan, np.nan, np.nan]

    return fit_params_full, fitted_curve, r2

# Optional alternative: fit A, B, C and gamma together
def fit_stretched_exponential_free_ab(t, g2, fit_params_single):
    """
    Fit all generalized exponential parameters.

    Parameters:
        t (array-like): One-dimensional delay times.
        g2 (array-like): One-dimensional correlation values.
        fit_params_single (array-like): Initial [A, B, C] estimates.

    Returns:
        tuple: (fit_params_stretched, fitted_curve_stretched, r2_stretched).
            - fit_params_stretched (numpy.ndarray): [A, B, C, gamma].
            - fitted_curve_stretched (numpy.ndarray): Model values at t.
            - r2_stretched (float): Coefficient of determination.
            Numerical failures return NaN parameters, curve and score.

    Notes:
        A and B are fitted again; gamma starts at 1.
        Optional alternative; not called by the main workflow.
    """
    # Initial guess for parameter gamma
    gamma0 = 1

    # Use A, B, C from the single fit and gamma0 as initial guesses.
    initial_params_stretched = [*fit_params_single, gamma0]

    # Fit the curve g2 = A + B * exp(-2 * (C * t)**gamma)
    # With constraints
    fit_params_stretched, fitted_curve_stretched, r2_stretched = (
        fit_model_with_constraints(
            t, g2, stretched_exponential, initial_params_stretched)
    )

    # Without constraints
    #fit_params_stretched, fitted_curve_stretched, r2_stretched = (
    #   fit_model(t, g2, stretched_exponential, initial_params_stretched)
    #)

    return fit_params_stretched, fitted_curve_stretched, r2_stretched

## Functions to fit a cumulants model to the provided data

# Function to fit C1 and C2 with A and B fixed
def fit_cumulants(t, g2, fit_params_single):
    """
    Fit C1 and C2 with A and B fixed from the single fit.

    Parameters:
        t (array-like): One-dimensional delay times.
        g2 (array-like): One-dimensional correlation values.
        fit_params_single (array-like): Single-fit parameters [A, B, C].

    Returns:
        tuple: (fit_params_full, fitted_curve, r2).
            - fit_params_full (list): [A, B, C1, C2].
            - fitted_curve (numpy.ndarray): Model values at t.
            - r2 (float): Coefficient of determination.
            Numerical failures return NaN parameters, curve and score.

    Notes:
        A and B remain fixed.
        Initial C1: single-fit rate. Initial C2: 0.05 * C1**2.
        C1 and C2 have zero lower bounds.
    """
    # Extract A, B, and C from the single exponential fit
    fit_params_single = np.asarray(fit_params_single, dtype=float)

    # Stop if the preceding fit did not provide three finite parameters.
    if (
        fit_params_single.shape != (3,)
        or not np.isfinite(fit_params_single).all()
    ):
        return [np.nan] * 4, np.full(len(t), np.nan), np.nan

    A, B, C1_single = fit_params_single
    
    # Initial guesses for C1 and C2
    C1_0 = C1_single
    C2_0 = 0.05 * (C1_0)**2  
    initial_params = [C1_0, C2_0]

    # Prepare the model with fixed A and B
    # Only C1 and C2 are exposed to the optimizer.
    def cumulants_model_refine(t, C1, C2):
        return cumulants_model(t, A, B, C1, C2)

    # Fit with constraints
    fit_params, fitted_curve, r2 = fit_model_with_constraints(
        t, g2, cumulants_model_refine, initial_params
    )

    # fit_params contains [C1, C2]
    # Restore the full parameter order [A, B, C1, C2].
    if np.isfinite(fit_params).all():
        C1_fit, C2_fit = fit_params
        fit_params_full = [A, B, C1_fit, C2_fit]
    else:
        fit_params_full = [np.nan, np.nan, np.nan, np.nan]

    return fit_params_full, fitted_curve, r2

# Optional alternative: fit A, B, C1 and C2 together
def fit_cumulants_free_ab(t, g2, fit_params_single):
    """
    Fit all second-order cumulants parameters.

    Parameters:
        t (array-like): One-dimensional delay times.
        g2 (array-like): One-dimensional correlation values.
        fit_params_single (array-like): Initial [A, B, C1] estimates.

    Returns:
        tuple: (fit_params_cumulants, fitted_curve_cumulants, r2_cumulants).
            - fit_params_cumulants (numpy.ndarray): [A, B, C1, C2].
            - fitted_curve_cumulants (numpy.ndarray): Model values at t.
            - r2_cumulants (float): Coefficient of determination.
            Numerical failures return NaN parameters, curve and score.

    Notes:
        A and B are fitted again.
        Initial C2: 0.05 * C1**2.
        Optional alternative; not called by the main workflow.
    """
    # Initial guess for parameter C2
    C2_0 = 0.05*fit_params_single[2]**2

    # Use the parameters A, B, C from the Single Exponential fit and C2 as initial guess
    initial_params_cumulants = [*fit_params_single, C2_0]

    # Fit the curve g2 = A + B * exp(-2C1 * t)*(1 + (1/2)* C2 * t**2)**2
    # With constraints
    fit_params_cumulants, fitted_curve_cumulants, r2_cumulants = (
        fit_model_with_constraints(
            t, g2, cumulants_model, initial_params_cumulants)
    )

    # Without constraints
    #fit_params_cumulants, fitted_curve_cumulants, r2_cumulants = (
    #   fit_model(t, g2, cumulants_model, initial_params_cumulants)
    #)

    return fit_params_cumulants, fitted_curve_cumulants, r2_cumulants

# Function to fit C = D * q^2 and return the slope and diagnostics
def fit_linear_model(q_squared_values, C_values):
    """
    Fit C = D * q**2 with zero intercept.

    Parameters:
        q_squared_values (array-like): Squared scattering vectors.
        C_values (array-like): Corresponding relaxation rates.

    Returns:
        tuple: (D, pearson_r, std_err).
            - D (float): Fitted slope; rate / q squared units.
            - pearson_r (float): Pearson correlation of rates and predictions.
            - std_err (float): Slope error estimated from fit covariance.
            Caught fitting errors return three NaN values.

    Notes:
        D represents diffusion under the corresponding physical assumptions.
        pearson_r is not R^2.
        Unidentifiable fits may produce nonfinite errors or correlations.
    """
    # Convert q_squared_values to float values
    q_squared_values = np.array(q_squared_values, dtype=np.float64)
    
    # Define the linear function to fit
    def linear_function(q_squared, D):
        return D * q_squared + 0  # Force y-intercept to be zero

    try:
       # Perform the linear fit
       fit_params, cov_matrix = curve_fit(
           linear_function, q_squared_values, C_values)
       
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
    
# Function to fit the power law C = D * q^n and return its parameters
def fit_exponential_model(q_values, C_values):
    """
    Fit the power law C = D * q**n.

    Parameters:
        q_values (array-like): Scattering vectors.
        C_values (array-like): Corresponding relaxation rates.

    Returns:
        tuple: (D, n, r2, std_err_D, std_err_n).
            - D (float): Power-law prefactor.
            - n (float): Dimensionless exponent.
            - r2 (float): Coefficient of determination.
            - std_err_D (float): Prefactor error from fit covariance.
            - std_err_n (float): Exponent error from fit covariance.
            Caught fitting errors return five NaN values.

    Notes:
        The function name is retained for compatibility.
        D units depend on n; D is not generally a diffusion coefficient.
        Unidentifiable fits may produce nonfinite parameter errors.
    """
    # Convert q_values and C_values to float values
    q_values = np.array(q_values, dtype=np.float64)
    C_values = np.array(C_values, dtype=np.float64)

    # Define the power-law function to fit
    def exponential_function(q, D, n):
        return D * q**n

    try:
        # Perform the power-law fit
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
    Calculate signed percentage differences from beamline parameters.

    Parameters:
        q_value (float): Scattering vector.
        beamline_params (dict or None): Reference fit parameters.
        code_params (dict or None): Program fit parameters.

    Returns:
        dict: q_value and percentage differences.
            - 'q_value' (float): Input scattering vector.
            - '*_variation_%' (float): Difference for each parameter.
            Unavailable differences are NaN.

    Notes:
        Parameters: 'Baseline', 'Beta', 'Relax_time', 'Gamma'.
        Difference: 100 * (program - reference) / reference.
        Zero, missing or nonfinite reference values produce NaN.
        Missing or nonfinite program values produce NaN.
        Each parameter is evaluated independently.
    """
    variation = {'q_value': q_value}

    # Calculate percentage variations for each parameter.
    # An unavailable parameter does not invalidate the remaining comparisons.
    for param in ['Baseline', 'Beta', 'Relax_time', 'Gamma']:
        difference_key = f'{param}_variation_%'
        variation[difference_key] = float('nan')

        try:
            hdf5_value = float(beamline_params[param])
            code_value = float(code_params[param])

            # Avoid division by zero and reject nonfinite parameter values.
            if (
                hdf5_value == 0
                or not np.isfinite(hdf5_value)
                or not np.isfinite(code_value)
            ):
                continue

            difference = (
                (code_value - hdf5_value) / hdf5_value
            ) * 100

            # Keep overflowed percentage differences as unavailable values.
            if np.isfinite(difference):
                variation[difference_key] = difference

        except (KeyError, TypeError, ValueError, OverflowError) as e:
            #print(f"### Error ###: Failed to calculate variation for q = {q_value} A^-1. Error: {e}")
            continue

    return variation

# Optional function to plot beamline parameter differences
def plot_fit_variations(variations_df):
    """
    Display percentage differences between program and beamline fits.

    Parameters:
        variations_df (pd.DataFrame): q_value and *_variation_% columns
            for Baseline, Beta, Relax_time and Gamma.

    Returns:
        None: Figure displayed using plt.show().

    Notes:
        Optional diagnostic plot; not called by the main workflow.
    """
    plt.figure(figsize=(12, 8))
    
    # Define the parameters to plot
    parameters = [
        'Baseline_variation_%',
        'Beta_variation_%',
        'Relax_time_variation_%',
        'Gamma_variation_%'
    ]
    labels = ['Baseline', 'Beta', 'Relax Time', 'Gamma']
    markers = ['o', 's', '^', 'D']
    
    # Plot each parameter's variation
    for param, label, marker in zip(parameters, labels, markers):
        plt.plot(
            variations_df['q_value'],
            variations_df[param],
            marker=marker,
            linestyle='-',
            label=label
        )

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

# Function to calculate inverse relaxation rate and the rate/q^2 quotient
def calculate_relaxation_and_diffusion(params, q_value):
    """
    Calculate inverse relaxation rate and the C/q^2 quotient.

    Parameters:
        params (array-like): Fit parameters.
            Position 2: C or C1, in s^-1.
        q_value (float): Scattering-vector magnitude, in angstrom^-1.

    Returns:
        tuple: (relax_time, diffusion_coef).
            - relax_time (float): Inverse rate, in s.
            - diffusion_coef (float): Converted C/q^2, in um^2/s.
            Unavailable results are NaN.

    Notes:
        C = 0: inverse rate is NaN; C/q^2 is zero for valid q.
        Negative or nonfinite rates produce NaN results.
        Invalid q affects only the C/q^2 calculation.
        Cumulants: the inverse rate is 1/C1.
        Diffusion interpretation requires the corresponding physical
        assumptions; C/q^2 is not generally a diffusion coefficient.
    """
    # Read C or C1 from the common parameter position.
    try:
        C = float(params[2])
    except (IndexError, TypeError, ValueError, OverflowError):
        return np.nan, np.nan

    # Nonfinite or negative rates cannot provide physical derived values.
    if not np.isfinite(C) or C < 0:
        return np.nan, np.nan

    relax_time = np.nan
    diffusion_coef = np.nan

    # Calculate the inverse rate only when C is strictly positive.
    if C > 0:
        with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
            relax_time = 1 / np.float64(C)                      # 1/C

        if not np.isfinite(relax_time):
            relax_time = np.nan

    # Validate q independently; an invalid q does not invalidate 1/C.
    try:
        q_value = float(q_value)
    except (TypeError, ValueError, OverflowError):
        return float(relax_time), np.nan

    if np.isfinite(q_value) and q_value > 0:
        with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
            q_squared = np.float64(q_value)**2                  # q^2

            # Require a finite, nonzero denominator.
            if np.isfinite(q_squared) and q_squared > 0:
                # Convert C/q^2 from angstrom^2/s to micrometre^2/s.
                diffusion_coef = (C / q_squared) * 1e-8         # C/q^2

        if not np.isfinite(diffusion_coef):
            diffusion_coef = np.nan

    return float(relax_time), float(diffusion_coef)

# Function to calculate the cumulants polydispersity index
def calculate_pdi(C1, C2):
    """
    Calculate the cumulants polydispersity index: C2/C1^2.

    Parameters:
        C1 (float): First cumulant, in inverse time units.
        C2 (float): Second cumulant, in inverse time squared units.

    Returns:
        float: Dimensionless PDI.
            Invalid parameters or nonfinite result: NaN.

    Notes:
        Require finite C1 > 0 and finite C2 >= 0.
        C2 = 0 gives PDI = 0.
    """
    # Convert scalar inputs before checking their values.
    try:
        C1 = float(C1)
        C2 = float(C2)
    except (TypeError, ValueError, OverflowError):
        return np.nan

    # A positive first cumulant is required to define the PDI.
    if not np.isfinite(C1) or C1 <= 0:
        return np.nan

    if not np.isfinite(C2) or C2 < 0:
        return np.nan

    # Divide twice to avoid overflowing or underflowing C1**2.
    with np.errstate(
        divide='ignore', invalid='ignore', over='ignore', under='ignore'
    ):
        pdi = (np.float64(C2) / np.float64(C1)) / np.float64(C1)

    return float(pdi) if np.isfinite(pdi) else np.nan

# Function to update parameter tracking
def update_parameter_tracking(parameter_data, model, parameter_value1, parameter_value2=None):
    """
    Record a complete set of finite parameters for averaging.

    Parameters:
        parameter_data (dict): Parameter accumulators.
        model (str): 'single', 'stretched' or 'cumulants'.
        parameter_value1 (float): Converted rate/q^2 value, in um^2/s.
        parameter_value2 (float or None): Gamma or PDI.
            Required for 'stretched' and 'cumulants'.

    Returns:
        bool: True if the record was included; False otherwise.

    Raises:
        ValueError: Unsupported model.

    Notes:
        Updates the input dictionary in place.
        Invalid values leave counters, sums and lists unchanged.
        Fit selection occurs in the caller.
    """
    if model not in ('single', 'stretched', 'cumulants'):
        raise ValueError(f"Unsupported model: {model!r}")

    parameters = parameter_data[model]

    # Validate all required values before changing counters or lists.
    try:
        diffusion_coef = float(parameter_value1)

        if not np.isfinite(diffusion_coef):
            return False

        if model in ('stretched', 'cumulants'):
            extra_value = float(parameter_value2)

            if not np.isfinite(extra_value):
                return False

    except (TypeError, ValueError, OverflowError):
        return False

    # Prepare the sums before updating the stored record.
    diffusion_sum = parameters['Diff_coef sum'] + diffusion_coef

    if not np.isfinite(diffusion_sum):
        return False

    if model == 'stretched':
        extra_key = 'Gamma'
    elif model == 'cumulants':
        extra_key = 'PDI'

    if model in ('stretched', 'cumulants'):
        extra_sum = parameters[f'{extra_key} sum'] + extra_value

        if not np.isfinite(extra_sum):
            return False

    # Update the counter and converted rate/q^2 sum.
    parameters['fit_count'] += 1
    parameters['Diff_coef sum'] = diffusion_sum

    # Update the additional parameter sum for the selected model.
    if model in ('stretched', 'cumulants'):
        parameters[f'{extra_key} sum'] = extra_sum

    # Store individual values for the standard-deviation calculation.
    parameters['Diff_coef values'].append(diffusion_coef)

    if model in ('stretched', 'cumulants'):
        parameters[f'{extra_key} values'].append(extra_value)

    return True

# Function to calculate average parameter values
def calculate_average_parameter_values(param_dict, model):
    """
    Calculate arithmetic means and population standard deviations.

    Parameters:
        param_dict (dict): Parameter accumulators.
        model (str): 'single', 'stretched' or 'cumulants'.

    Returns:
        dict: Parameter means and standard deviations.
            - 'Diff_coef av', 'Diff_coef std' (float): All models.
            - 'Gamma av', 'Gamma std' (float): Stretched model.
            - 'PDI av', 'PDI std' (float): Cumulants model.
            Empty selections return NaN values.

    Raises:
        ValueError: Unsupported model, negative count or inconsistent
            parameter-list lengths.

    Notes:
        Means use the stored sums and fit_count.
        Standard deviations use ddof=0; one value gives zero.
        Nonfinite stored data produce NaN for the affected parameter.
        The deviation describes the spread of included values,
        not the uncertainty of an individual fit.
    """
    if model not in ('single', 'stretched', 'cumulants'):
        raise ValueError(f"Unsupported model: {model!r}")

    parameters = param_dict[model]
    fit_count = parameters['fit_count']

    if fit_count < 0:
        raise ValueError("fit_count cannot be negative.")

    # Prepare all expected output keys, including for empty selections.
    parameter_names = ['Diff_coef']

    if model == 'stretched':
        parameter_names.append('Gamma')
    elif model == 'cumulants':
        parameter_names.append('PDI')

    average_parameters = {
        f'{parameter} {statistic}': np.nan
        for parameter in parameter_names
        for statistic in ('av', 'std')
    }

    # No included fits: means and standard deviations remain NaN.
    if fit_count == 0:
        return average_parameters

    # Apply the same checks and statistics to each tracked parameter.
    def parameter_summary(parameter):
        values = np.asarray(
            parameters[f'{parameter} values'],
            dtype=float
        )
        total = float(parameters[f'{parameter} sum'])

        # The sum, count and list must describe the same included records.
        if values.ndim != 1 or values.size != fit_count:
            raise ValueError(
                f"Inconsistent {parameter} records "
                f"for model {model!r}."
            )

        # Do not silently change the selection by dropping stored values.
        if not np.isfinite(total) or not np.isfinite(values).all():
            return np.nan, np.nan

        with np.errstate(over='ignore', invalid='ignore'):
            average = total / fit_count
            std = np.std(values, ddof=0)

        if not np.isfinite(average):
            average = np.nan
        if not np.isfinite(std):
            std = np.nan

        return float(average), float(std)

    # Calculate the average Diff_coef value.
    # Calculate the standard deviation for Diff_coef.
    average_parameters['Diff_coef av'], average_parameters['Diff_coef std'] = (
        parameter_summary('Diff_coef')
    )

    if model == 'stretched':
        # Calculate the average Gamma value.
        # Calculate the standard deviation for Gamma.
        average_parameters['Gamma av'], average_parameters['Gamma std'] = (
            parameter_summary('Gamma')
        )

    elif model == 'cumulants':
        # Calculate the average PDI value.
        # Calculate the standard deviation for PDI.
        average_parameters['PDI av'], average_parameters['PDI std'] = (
            parameter_summary('PDI')
        )

    return average_parameters

#-----------------------------------------------------------------------#
#-------------------- Result Saving and Reporting ----------------------#
#-----------------------------------------------------------------------#

# Function to initialize the analysis record for one input file
def initialize_analysis_record(file_path, base_name, selected_synchrotron,
                               data_type, r2_threshold, time_range,
                               q_values_by_column, selected_columns,
                               compare_with_beamline=False):
    """
    Initialize provenance for one input file and execution.

    Parameters:
        file_path (str or os.PathLike): Original HDF5 path.
        base_name (str): Output filename prefix.
        selected_synchrotron (str): Input format.
        data_type (str or None): Data representation.
        r2_threshold (float): Threshold for parameter averages.
        time_range (tuple or None): Requested time bounds, in seconds.
            None: full available interval.
        q_values_by_column (dict): Input column names and q values.
        selected_columns (iterable): Correlation columns kept for analysis.
        compare_with_beamline (bool): Requested reference comparison.
            Default: False. Supported only for Sirius 3.

    Returns:
        dict: Settings, UTC identifier and empty curve-record list.
    """
    # Use UTC with microseconds to identify this file's analysis.
    started_at = datetime.now(timezone.utc)
    selected_columns = list(selected_columns)

    return {
        'run_id': started_at.strftime('%Y%m%dT%H%M%S%fZ'),
        'started_at': started_at.isoformat(),
        'software_version': __version__,
        'input_file': os.path.basename(file_path),
        'input_path': os.path.abspath(file_path),
        'base_name': base_name,
        'input_format': selected_synchrotron,
        'data_type': data_type,
        'r2_threshold': float(r2_threshold),
        'compare_with_beamline': bool(compare_with_beamline),
        'time_range': time_range,
        'selected_q': [float(q_values_by_column[col])
                       for col in selected_columns],
        'excluded_q': [float(q) for col, q in q_values_by_column.items()
                       if col not in selected_columns],
        'curves': [],
    }

# Function to record the data actually supplied to the fits
def record_analysis_curve(analysis_record, q_value, t=None, r2_values=None):
    """
    Append the fitted interval and numerical status of one q curve.

    Parameters:
        analysis_record (dict): Current file's analysis record.
        q_value (float): Scattering-vector magnitude, in angstrom^-1.
        t (array-like or None): Cleaned times supplied to the models.
            None: curve skipped before fitting.
        r2_values (tuple or None): Single, stretched and cumulants R^2.

    Returns:
        None: Updates analysis_record in place.

    Notes:
        Skipped curves have zero fitted points and NaN time bounds.
        Numerical completion does not imply acceptance for averages.
    """
    # Keep skipped curves in the record without inventing time bounds.
    if t is None:
        n_points, t_min, t_max = 0, np.nan, np.nan
        scores = (np.nan, np.nan, np.nan)
        status = 'skipped_too_few_points'
    else:
        times = np.asarray(t, dtype=float)
        n_points = int(times.size)
        t_min, t_max = float(times.min()), float(times.max())
        scores = tuple(float(value) for value in r2_values)
        status = ('all_fit_scores_finite' if np.isfinite(scores).all()
                  else 'one_or_more_fit_scores_nonfinite')

    r2_single, r2_stretched, r2_cumulants = scores

    analysis_record['curves'].append({
        'q_A^-1': float(q_value),
        'n_fit_points': n_points,
        't_min_s': t_min,
        't_max_s': t_max,
        'status': status,
        'R2_single': r2_single,
        'R2_stretched': r2_stretched,
        'R2_cumulants': r2_cumulants,
    })

# Function to save the analysis record without overwriting previous runs
def save_analysis_record(analysis_record, directory):
    """
    Save settings and per-q records as readable text and a TSV table.

    Parameters:
        analysis_record (dict): Completed settings and curve records.
        directory (str or os.PathLike): Current file's output directory.

    Returns:
        str: Saved analysis-record path.

    Raises:
        AnalysisStopped: Record cannot be saved or already exists.

    Notes:
        UTF-8 text; LF line endings in both the header and table.
        Legacy pandas: retry with 'line_terminator'.
    """
    record = analysis_record

    output_path = os.path.join(
        directory,
        f"{record['base_name']}_analysis_{record['run_id']}.dat"
    )

    settings = {
        'Run ID (UTC)': record['run_id'],
        'Started at (UTC)': record['started_at'],
        'Recorded at (UTC)': datetime.now(timezone.utc).isoformat(),
        'Software version': record['software_version'],
        'Original filename': record['input_file'],
        'Original path': record['input_path'],
        'Output base name': record['base_name'],
        'Input format': record['input_format'],
        'Data representation': record['data_type'] or 'Not applicable',
        'R^2 threshold': record['r2_threshold'],
        'Compare with beamline fits': record.get('compare_with_beamline', False),
        'Requested interval (s)': (
            record['time_range']
            if record['time_range'] is not None
            else 'Full available interval'
        ),
        'Selected q (angstrom^-1)': record['selected_q'],
        'Excluded q (angstrom^-1)': record['excluded_q'],
    }

    try:
        # Exclusive creation protects records from previous executions.
        with open(output_path, 'x', encoding='utf-8', newline='') as file:
            file.write('# XPCS_FitLab analysis record\n')

            for name, value in settings.items():
                file.write(f'# {name}: {value}\n')

            file.write(
                '# Unit assumption: input time in s and q in angstrom^-1.\n'
                '# Input units are not automatically validated or converted.\n'
                '# Rate/q^2 conversion: angstrom^2/s to um^2/s, factor 1e-8.\n'
                '# Diffusion interpretation requires an appropriate physical model.\n'
                '# Average selection: finite R^2 strictly above the threshold; '
                'finite derived parameters also required.\n'
                '# Cleaning: finite time-g2 pairs; remove a leading zero time; '
                'at least 8 valid points required.\n'
                '# Models: single exponential, stretched exponential, cumulants.\n'
                '# Baseline A and amplitude B: fitted in the single exponential; '
                'fixed in the stretched exponential and cumulants fits.\n'
                '# Per-q times and counts describe data supplied to the models.\n'
                '# Numerical fit status is independent of the R^2 threshold.\n'
                '# NaN: unavailable value.\n\n'
            )

            try:
                pd.DataFrame(record['curves']).to_csv(
                    file, sep='\t', index=False, na_rep='NaN',
                    lineterminator='\n'
                )
            except TypeError as error:
                # Retry only when legacy pandas rejects this keyword.
                if "unexpected keyword argument 'lineterminator'" not in str(error):
                    raise

                pd.DataFrame(record['curves']).to_csv(
                    file, sep='\t', index=False, na_rep='NaN',
                    line_terminator='\n'
                )

    except OSError as error:
        raise AnalysisStopped(
            f"Could not save analysis record '{output_path}'. Reason: {error}"
        ) from None

    return output_path

# Function to update success and failure counters for one q series
def record_fit_status(counters, file_name, q_value, r2_values):
    """
    Record the numerical fit status of one q series.

    Parameters:
        counters (dict): Processing counters and record lists.
        file_name (str): Input filename.
        q_value (float): Scattering-vector magnitude, in angstrom^-1.
        r2_values (array-like): R^2 values from the three model fits.

    Returns:
        None: Updates counters in place.

    Notes:
        Call once per q series, after all three fits.
        All finite scores: increment success.
        Any nonfinite score: increment failure once.
        Selection for parameter averages is evaluated separately.
    """
    # A finite score, even below the threshold, is not a numerical failure.
    if np.isfinite(r2_values).all():
        counters['success'] += 1
    else:
        # Record this curve once, regardless of how many model fits failed.
        counters['failure'] += 1
        counters['failed_files'].append(
            f"{file_name} - q = {q_value} A^-1"
        )

# Function to add the data to the table
def add_data_to_table(table_data, base_name, q_value, params, relax_time, diffusion_coef, r2):
    """
    Append one result row to the supplied table.

    Parameters:
        table_data (list): Destination list of rows.
        base_name (str): Input filename identifier.
        q_value (float): Scattering-vector magnitude.
        params (array-like): Parameters to include in the row.
        relax_time (float): Inverse relaxation rate.
        diffusion_coef (float): Converted rate/q^2 value.
        r2 (float): R^2 score.

    Returns:
        None: Input list updated in place.

    Notes:
        Row: [base_name, q_value, *params, relax_time, diffusion_coef, r2].
        Single params: [A, B, C].
        Stretched params: [A, B, C, gamma].
        Cumulants params: [A, B, C1, PDI].
        NaN values are preserved for reporting failed fits.
    """
    table_row = [base_name, q_value, *params, relax_time, diffusion_coef, r2]
    table_data.append(table_row)

# Function to write data to .dat file
def write_dat_file(output_path, q_value, t, g2, A_single, B_single, C_single, 
                   A_stretched, B_stretched, C_stretched, gamma,
                   r2_single, r2_stretched, relax_time_single, relax_time_stretched, 
                   diffusion_coef_single, diffusion_coef_stretched,
                   A_cumulants, B_cumulants, C1_cumulants, C2_cumulants, r2_cumulants, 
                   relax_time_cumulants, diffusion_coef_cumulants):
    """
    Write fit metadata, experimental data and model curves.

    Parameters:
        output_path (str or os.PathLike): Output .dat path.
        q_value (float): Scattering vector, in angstrom^-1.
        t (array-like): One-dimensional delay times, in s.
        g2 (array-like): Corresponding correlation values.
        A_single, B_single, C_single (float): Single-fit parameters.
        A_stretched, B_stretched, C_stretched, gamma (float):
            Generalized exponential parameters.
        r2_single, r2_stretched (float): R^2 scores.
        relax_time_single, relax_time_stretched (float):
            Inverse rates, in s.
        diffusion_coef_single, diffusion_coef_stretched (float):
            Converted rate/q^2 values, in um^2/s.
        A_cumulants, B_cumulants, C1_cumulants, C2_cumulants (float):
            Cumulants parameters.
        r2_cumulants (float): Cumulants R^2 score.
        relax_time_cumulants (float): Inverse C1, in s.
        diffusion_coef_cumulants (float): Converted C1/q^2, in um^2/s.

    Returns:
        None: Output file written.

    Raises:
        ValueError: t and g2 are not matching one-dimensional series.
        OSError: Output file cannot be written.

    Notes:
        Six tab-separated columns:
        t, g2, std_g2_series, single fit, stretched fit, cumulants fit.
        std_g2_series: std(g2, ddof=0), repeated on every row.
        Metadata lines start with '#'.
        Existing files are overwritten.
    """
    # Convert inputs to arrays for positional indexing.
    t = np.asarray(t, dtype=float)
    g2 = np.asarray(g2, dtype=float)

    # Validate row alignment before opening the output file.
    if t.ndim != 1 or g2.ndim != 1 or t.shape != g2.shape:
        raise ValueError(
            "t and g2 must be matching one-dimensional series.")

    # Calculate a descriptive deviation across the series, once per file.
    with np.errstate(over='ignore', invalid='ignore'):
        std_g2_series = np.std(g2, ddof=0) if g2.size else np.nan

    if not np.isfinite(std_g2_series):
        std_g2_series = np.nan

    # Calculate PDI using the same checks as the summary table.
    PDI_cumulants = calculate_pdi(C1_cumulants, C2_cumulants)

    # Reuse the same model definitions used during fitting.
    # Failed fits can retain NaN curves in the exported results.
    with np.errstate(
        over='ignore',
        invalid='ignore',
        divide='ignore'
    ):
        fitted_curve_single = single_exponential(
            t, A_single, B_single, C_single
        )
        fitted_curve_stretched = stretched_exponential(
            t, A_stretched, B_stretched, C_stretched, gamma
        )
        fitted_curve_cumulants = cumulants_model(
            t, A_cumulants, B_cumulants, C1_cumulants, C2_cumulants
        )

    # Write UTF-8 text with explicit LF line endings.
    with open(
        output_path, 'w', encoding='utf-8', newline=''
    ) as dat_file:
        dat_file.write(f"# q = {q_value} A^-1\n")

        dat_file.write("### Fit model: Single Exponential ###\n")
        dat_file.write(
            f"#Fit parameters: baseline = {A_single}, "
            f"beta = {B_single}, C (s^-1) = {C_single}\n"
        )
        dat_file.write(f"#R^2: {r2_single}\n")
        dat_file.write(
            f"#Derived parameters: 1/C (s) = {relax_time_single}, "
            f"C/q^2 (um^2/s) = {diffusion_coef_single}\n"
        )

        dat_file.write("### Fit model: Stretched Exponential ###\n")
        dat_file.write(
            f"#Fit parameters: baseline = {A_stretched}, "
            f"beta = {B_stretched}, C (s^-1) = {C_stretched}, "
            f"gamma = {gamma}\n"
        )
        dat_file.write(f"#R^2: {r2_stretched}\n")
        dat_file.write(
            f"#Derived parameters: 1/C (s) = {relax_time_stretched}, "
            f"C/q^2 (um^2/s) = {diffusion_coef_stretched}\n"
        )

        dat_file.write("### Fit model: Cumulants Model ###\n")
        dat_file.write(
            f"#Fit parameters: baseline = {A_cumulants}, "
            f"beta = {B_cumulants}, C1 (s^-1) = {C1_cumulants}, "
            f"C2 (s^-2) = {C2_cumulants}, PDI = {PDI_cumulants}\n"
        )
        dat_file.write(f"#R^2: {r2_cumulants}\n")
        dat_file.write(
            f"#Derived parameters: 1/C1 (s) = {relax_time_cumulants}, "
            f"C1/q^2 (um^2/s) = {diffusion_coef_cumulants}\n"
        )

        dat_file.write(
            "# std_g2_series: std(g2, ddof=0) across delay times; "
            "not a per-point experimental uncertainty.\n"
        )
        dat_file.write(
            "#delay time (s)\tg2\tstd_g2_series\t"
            "Fitted g2 Single\tFitted g2 Stretched\t"
            "Fitted g2 Cumulants\n"
        )

        # Write six tab-separated values for each observation.
        for i in range(len(t)):
            row_str = (
                f"{t[i]}\t{g2[i]}\t{std_g2_series}\t"
                f"{fitted_curve_single[i]}\t"
                f"{fitted_curve_stretched[i]}\t"
                f"{fitted_curve_cumulants[i]}\n"
            )
            dat_file.write(row_str)

# Function to save figures to a multipage PDF
def save_figures_to_pdf(figures, output_path):
    """
    Save figures as consecutive PDF pages.

    Parameters:
        figures (list or tuple): Matplotlib figures, in page order.
        output_path (str or os.PathLike): Output PDF path.

    Returns:
        bool: True if saved; False if saving failed.

    Notes:
        Figures are closed after the save attempt.
        Existing output files are overwritten.
        Caught errors are printed with their technical details.
    """
    # Keep the figure references available for cleanup.
    figures = tuple(figures)

    try:
        with PdfPages(output_path) as pdf:
            # Save each figure and its subplots on a separate page.
            for figure in figures:
                pdf.savefig(figure)

        return True

    except (OSError, ValueError, RuntimeError) as error:
        print(f"Warning: Could not save PDF '{output_path}'. "
              f"Reason: {error}")
        return False

    finally:
        # Close only the figures passed to this function.
        for figure in figures:
            plt.close(figure)

# Function to save beamline fit comparisons
def save_fit_variations(fit_variations, output_path):
    """
    Save percentage differences from beamline fit parameters.

    Parameters:
        fit_variations (list): Comparison dictionaries.
        output_path (str or os.PathLike): Output TSV .dat path.

    Returns:
        bool: True if written; False if unnecessary or saving failed.

    Notes:
        Empty results: no new file.
        An existing report is rewritten with headers when results are empty.
        Caught errors are printed with their technical details.
    """
    # Clear an old report if this run produced no comparisons.
    if not fit_variations and not os.path.isfile(output_path):
        return False

    columns = ['q_value',
               'Baseline_variation_%',
               'Beta_variation_%',
               'Relax_time_variation_%',
               'Gamma_variation_%']

    try:
        # Keep a readable header even when there are no comparison rows.
        variations_df = pd.DataFrame(fit_variations, columns=columns)
        save_dataframe_to_tsv(variations_df, output_path)
        return True

    except (OSError, ValueError) as error:
        print(f"Warning: Could not save beamline comparisons '{output_path}'. "
              f"Reason: {error}")
        return False

# Function to print processing counters and save the low-R^2 report
def print_summary(counters, directory=None, r2_threshold=None):

    """
    Print processing counters and optionally save low-R^2 records.

    Parameters:
        counters (dict): Processing counters and record lists.
        directory (str or os.PathLike or None): Report destination.
        r2_threshold (float or None): Selection threshold.

    Returns:
        None: Summary printed; report written when a directory is supplied.

    Raises:
        OSError: Report file cannot be written.

    Notes:
        'failure' counts q series; 'below_threshold' counts model fits.
        'success' counts q series with three finite fit R^2 values.
        Excluded and skipped curves are counted separately.
        The report is rewritten even when its record list is empty.
    """
    # Count curves submitted to the three model fits.
    attempted_q = counters['success'] + counters['failure']

    print(f"Total HDF5 files: {counters['hdf5_files']}")
    print(f"Files skipped before analysis: {counters['invalid_data']}")
    print(f"Recognized input q series: {counters['q_values']}")
    print(f"q series excluded by q selection: {counters['excluded_q']}")
    print(f"q series skipped before fitting: {counters['skipped_q']}")
    print(f"q series submitted to fitting: {attempted_q}")
    print(f"q series with three finite fit scores: {counters['success']}")
    print(f"q series with at least one failed fit: {counters['failure']}")
    print(f"Model fits at or below R^2 threshold: {counters['below_threshold']}")

    # Print the filenames and reasons recorded before analysis.
    if counters['invalid_files']:
        print("Files skipped before analysis:")
        for file_name in counters['invalid_files']:
            print(file_name)

    # Print one record per q series with a numerical fit failure.
    if counters['failed_files']:
        print("Series with fitting failures:")
        for file_name in counters['failed_files']:
            print(file_name)

    # Save low-R^2 records to a separate report file.
    # Rewrite empty reports too, so a previous run cannot leave stale records.
    if directory:
        below_threshold_path = os.path.join(
            directory, "below_R2_threshold_fits.dat")

        with open(below_threshold_path,'w',encoding='utf-8',
                  newline='') as report_file:
            report_file.write("# Model fits at or below R^2 threshold\n")

            if r2_threshold is not None:
                report_file.write(f"# R2_threshold = {r2_threshold}\n")

            report_file.write("# Selection requires R^2 strictly greater "
                "than the threshold.\n")
            report_file.write(
                "# Listed fits did not pass this selection criterion.\n")
            report_file.write("# Format: filename - q - model - R2\n")

            for file_name in counters['below_threshold_files']:
                report_file.write(f"{file_name}\n")

        print(f"Detailed low-R^2 fit list saved to: "
              f"{below_threshold_path}")

    elif counters['below_threshold'] != 0:
        print("Detailed low-R^2 fit list was not saved because "
            "no output directory was provided.")

# Function to generate fit results tables
def generate_fit_results_tables(directory, table_data_single, 
                                table_data_stretched, table_data_cumulants):
    """
    Write the three space-padded fit-result tables.

    Parameters:
        directory (str or os.PathLike): Output directory.
        table_data_single (list): Rows with 8 elements.
        table_data_stretched (list): Rows with 9 elements.
        table_data_cumulants (list): Rows with 9 elements.

    Returns:
        None: Three .dat files written.

    Raises:
        ValueError: A row has an unexpected number of elements.
        OSError: An output file cannot be written.

    Notes:
        Row order follows add_data_to_table.
        The cumulants table contains PDI, not C2.
        Empty tables retain their headers.
        Field widths are minimum widths; long values are preserved.
        Existing files are overwritten.
    """
    # Validate all row lengths before opening any output file.
    for model, rows, expected_columns in (
        ('single', table_data_single, 8),
        ('stretched', table_data_stretched, 9),
        ('cumulants', table_data_cumulants, 9)
    ):
        for row_index, row in enumerate(rows, start=1):
            if len(row) != expected_columns:
                raise ValueError(
                    f"{model} row {row_index}: expected "
                    f"{expected_columns} columns, "
                    f"received {len(row)}."
                )

    table_output_path_single = os.path.join(directory, 
                                            "fit_results_single.dat")
    table_output_path_stretched = os.path.join(directory, 
                                               "fit_results_stretched.dat")
    table_output_path_cumulants = os.path.join(directory, 
                                               "fit_results_cumulants.dat")

    with open(
        table_output_path_single, 'w', encoding='utf-8', newline=''
    ) as table_file_single, open(
        table_output_path_stretched, 'w', encoding='utf-8', newline=''
    ) as table_file_stretched, open(
        table_output_path_cumulants, 'w', encoding='utf-8', newline=''
    ) as table_file_cumulants:

        # Write header for Single Exponential
        header_single = (
            "{:^26} {:^13} {:^23} {:^24} "
            "{:^24} {:^26} {:^25} {:^24}\n"
        ).format(
            "Filename", "q (A^-1)", "Baseline", "beta",
            "C (s^-1)", "1/C (s)",
            "C/q^2 (um^2/s)", "R^2"
        )
        table_file_single.write(header_single)

        # Write header for Stretched Exponential
        header_stretched = (
            "{:^26} {:^13} {:^23} {:^24} {:^24} "
            "{:^24} {:^26} {:^25} {:^24}\n"
        ).format(
            "Filename", "q (A^-1)", "Baseline", "beta",
            "C (s^-1)", "Gamma", "1/C (s)",
            "C/q^2 (um^2/s)", "R^2"
        )
        table_file_stretched.write(header_stretched)

        # Write header for Cumulants
        header_cumulants = (
            "{:^26} {:^13} {:^23} {:^24} {:^24} "
            "{:^24} {:^26} {:^25} {:^24}\n"
        ).format(
            "Filename", "q (A^-1)", "Baseline", "beta",
            "C1 (s^-1)", "PDI", "1/C1 (s)",
            "C1/q^2 (um^2/s)", "R^2"
        )
        table_file_cumulants.write(header_cumulants)

        # Write table data for Single Exponential
        for row_single in table_data_single:
            row_str_single = (
                "{:<26} {:<13} {:<23} {:<24} "
                "{:<24} {:<26} {:<25} {:<24}\n"
            ).format(*row_single)
            table_file_single.write(row_str_single)

        # Write table data for Stretched Exponential
        for row_stretched in table_data_stretched:
            row_str_stretched = (
                "{:<26} {:<13} {:<23} {:<24} {:<24} "
                "{:<24} {:<26} {:<25} {:<24}\n"
            ).format(*row_stretched)
            table_file_stretched.write(row_str_stretched)

        # Write table data for Cumulants
        for row_cumulants in table_data_cumulants:
            row_str_cumulants = (
                "{:<26} {:<13} {:<23} {:<24} {:<24} "
                "{:<24} {:<26} {:<25} {:<24}\n"
            ).format(*row_cumulants)
            table_file_cumulants.write(row_str_cumulants)

    if table_data_single or table_data_stretched or table_data_cumulants:
        print("Fit results tables generated successfully.")
    else:
        print("No fit results were obtained. Output tables contain headers only.")

#-----------------------------------------------------------------------#
#--------------------- Plotting and Visualization ----------------------#
#-----------------------------------------------------------------------#

# Function to identify the analysis on both PDF pages
def add_analysis_details(figures, analysis_record):
    """
    Add the original filename, settings and run identifier to figures.

    Parameters:
        figures (list or tuple): Figures exported to the PDF.
        analysis_record (dict): Current file's analysis record.

    Returns:
        None: Adds a footer and reserves space below the subplots.
    """
    record = analysis_record

    if record['time_range'] is None:
        interval = 'full available interval'
    else:
        t_min, t_max = record['time_range']
        interval = f'{t_min:.6g} to {t_max:.6g} s'

    # Wrap long filenames; the full path remains in the text record.
    footer = '\n'.join([
        textwrap.fill(
            f"Input: {record['input_file']}", width=135
        ),
        textwrap.fill(
            f"Format: {record['input_format']} | "
            f"Data: {record['data_type'] or 'not applicable'} | "
            f"R^2 threshold: {record['r2_threshold']:.6g} (strict >) | "
            f"Requested t: {interval} | "
            f"Selected q: {len(record['selected_q'])}",
            width=135
        ),
        f"XPCS_FitLab {record['software_version']} | "
        f"Run (UTC): {record['run_id']}",
    ])

    for figure in figures:
        # Preserve the existing layout except for the bottom margin.
        figure.subplots_adjust(
            bottom=max(figure.subplotpars.bottom, 0.10))

        figure.text(0.05, 0.012, footer,
                    fontsize=9, ha='left', va='bottom',
                    linespacing=1.2, parse_math=False)

# Function to plot experimental data and fitted curves
def plot_data_and_curves(ax, t, g2, baseline, Beta, fitted_curve, 
                         q_value, r2_value, lines_labels_dict, color, 
                         linestyle, model_type):
    """
    Plot normalized correlations and fitted curves.

    Parameters:
        ax (matplotlib.axes.Axes): Target subplot.
        t (array-like): Delay times; one-dimensional.
        g2 (array-like): Experimental correlation values.
        baseline (float): Fitted baseline A.
        Beta (float): Fitted amplitude B.
        fitted_curve (array-like): Fitted correlation values.
        q_value (float): Scattering vector.
        r2_value (float): R^2 of the correlation fit.
        lines_labels_dict (dict): Line and label lists for each model.
        color (str or tuple): Matplotlib color.
        linestyle (str): Fitted-curve line style.
        model_type (str): 'Single', 'Stretched', or 'Cumulants'.

    Returns:
        None

    Notes:
        Normalize both curves as (g2 - A) / B.
        Require matching one-dimensional arrays and positive finite times.
        Skip undefined normalization or entirely invalid experimental data.
        Nonfinite normalized values appear as gaps.
    """
    t = np.asarray(t, dtype=float)
    g2 = np.asarray(g2, dtype=float)
    fitted_curve = np.asarray(fitted_curve, dtype=float)

    # Check alignment before plotting or changing the legend lists.
    if t.ndim != 1 or g2.ndim != 1 or fitted_curve.ndim != 1:
        raise ValueError("t, g2, and fitted_curve must be one-dimensional.")
    if not (len(t) == len(g2) == len(fitted_curve)):
        raise ValueError("t, g2, and fitted_curve must have equal lengths.")

    model = model_type.lower()
    if model not in ('single', 'stretched', 'cumulants'):
        raise ValueError("Unsupported correlation model.")

    # Check all legend keys before adding any lines.
    exp_lines = lines_labels_dict[f"exp_lines_{model}"]
    exp_labels = lines_labels_dict[f"exp_labels_{model}"]
    fit_lines = lines_labels_dict[f"fit_lines_{model}"]
    fit_labels = lines_labels_dict[f"fit_labels_{model}"]

    # Normalization is undefined for zero amplitude or nonfinite A/B.
    if not np.isfinite(baseline) or not np.isfinite(Beta) or Beta == 0:
        print(f"Skipping {model} plot: invalid normalization parameters.")
        return

    # A logarithmic time axis requires positive finite times.
    valid_time = np.isfinite(t) & (t > 0)
    t = t[valid_time]

    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        normalized_g2 = (g2[valid_time] - baseline) / Beta
        normalized_fit = (fitted_curve[valid_time] - baseline) / Beta

    if not np.any(np.isfinite(normalized_g2)):
        print(f"Skipping {model} plot: no valid experimental points.")
        return

    # Keep gaps instead of connecting across invalid curve values.
    normalized_g2 = np.where(
        np.isfinite(normalized_g2), normalized_g2, np.nan
    )
    normalized_fit = np.where(
        np.isfinite(normalized_fit), normalized_fit, np.nan
    )

    # Plot experimental data points.
    line_exp = ax.semilogx(t, normalized_g2, 'o', color=color)[0]

    # Plot the fitted curve.
    line_fit = ax.semilogx(
        t, normalized_fit, color=color, linestyle=linestyle)[0]

    # Add labels to the legend lists based on the model type.
    exp_lines.append(line_exp)
    exp_labels.append(f"q = {q_value:.6f}")
    fit_lines.append(line_fit)
    fit_labels.append(f"R^2: {r2_value:.3f}")

# Function to fit and plot a specified model
def fit_and_plot_model(q_values, relax_rate_values, ax, colors, label, 
                       model_type='linear'):
    """
    Fit and plot relaxation rate versus q^2 or q.

    Parameters:
        q_values (array-like): Scattering vectors; one-dimensional.
        relax_rate_values (array-like): Corresponding relaxation rates.
        ax (matplotlib.axes.Axes): Target subplot.
        colors (list): One color per input q value.
        label (str): Fitted-line label.
        model_type (str, optional): 'linear' or 'exponential'. Default: 'linear'.

    Returns:
        tuple: (coefficient, n, r_metric, std_err).
            - coefficient (float): D_app for linear; K for power law.
            - n (float): Power-law exponent; NaN for the linear model.
            - r_metric (float): Pearson r for linear; R^2 for power law.
            - std_err (float or tuple): Slope error or (K error, n error).
            Insufficient data returns four NaN values.

    Notes:
        'linear': C = D_app * q^2; 'exponential': C = K * q^n.
        The local variable D stores either coefficient internally.
        Keep the historical 'exponential' identifier for compatibility.
        Ignore nonfinite or nonpositive q/rate values; preserve their colors.
        Minimum points: two for linear; three for power law.
        These counts do not guarantee an identifiable fit.
        Keep input units; no diffusion-unit conversion is applied here.
    """
    if model_type not in ('linear', 'exponential'):
        raise ValueError("model_type must be 'linear' or 'exponential'.")

    q_values = np.asarray(q_values, dtype=float)
    relax_rate_values = np.asarray(relax_rate_values, dtype=float)
    colors = list(colors)

    # Check alignment before filtering data and colors together.
    if q_values.ndim != 1 or relax_rate_values.ndim != 1:
        raise ValueError(
            "q_values and relax_rate_values must be one-dimensional.")
    if not (len(q_values) == len(relax_rate_values) == len(colors)):
        raise ValueError(
            "q_values, relax_rate_values, and colors must have equal lengths.")

    # Keep positive finite q values and relaxation rates.
    valid_mask = (
        np.isfinite(q_values)
        & np.isfinite(relax_rate_values)
        & (q_values > 0)
        & (relax_rate_values > 0))

    # Use q^2 for the linear fit and q for the power-law fit.
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        x_values = q_values**2 if model_type == 'linear' else q_values

    valid_mask &= np.isfinite(x_values) & (x_values > 0)
    x_values = x_values[valid_mask]
    relax_rate_values = relax_rate_values[valid_mask]
    colors = [color for color, keep in zip(colors, valid_mask) if keep]

    # Plot valid data even when there are too few points to fit.
    for x, y, color in zip(x_values, relax_rate_values, colors):
        ax.plot(x, y, 'o', color=color, markersize=10)

    # Set axes safely, including the empty-data case.
    ax.set_xlim(0, np.max(x_values) * 1.1 if x_values.size else 1)
    ax.set_ylim(0, np.max(relax_rate_values) * 1.1 if x_values.size else 1)

    # Require two points for the linear fit and three for the power-law fit.
    min_points = 2 if model_type == 'linear' else 3

    if len(x_values) < min_points:
        ax.text(0.5, 0.5, "Not enough valid fits",
                transform=ax.transAxes, ha='center', va='center', fontsize=12)
        print(f"Skipping {model_type} model: "
              f"not enough valid points ({len(x_values)} available).")
        return np.nan, np.nan, np.nan, np.nan

    if model_type == 'linear':
        D, r_metric, std_err = fit_linear_model(x_values, relax_rate_values)
        n = np.nan
    else:
        D, n, r_metric, std_err_D, std_err_n = fit_exponential_model(
            x_values, relax_rate_values)
        std_err = (std_err_D, std_err_n)

    # If the fit failed internally, do not try to draw a fit line.
    if not np.isfinite(D) or (
        model_type == 'exponential' and not np.isfinite(n)):
        ax.text(0.5, 0.5, "Fit failed",
                transform=ax.transAxes, ha='center', va='center', fontsize=12)
        return D, n, r_metric, std_err

    # Generate the fit line using the fitted values.
    x_fit = np.linspace(
        0 if model_type == 'linear' else np.min(x_values),np.max(x_values),100)

    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        fit_line = (D * x_fit
            if model_type == 'linear'
            else D * x_fit**n)

    # Plot the fit line only if it is finite.
    if np.all(np.isfinite(fit_line)):
        ax.plot(x_fit, fit_line, linestyle='-', color='black', label=label)
    else:
        ax.text(0.5, 0.5, "Fit curve not finite",
            transform=ax.transAxes, ha='center', va='center', fontsize=12)

    return D, n, r_metric, std_err

# Function to label and annotate relaxation-rate fits
def annotate_rate_fit(ax, coefficient, exponent, r_metric, std_err,
                      model_type='linear'):
    """
    Label and annotate a relaxation-rate fit.

    Parameters:
        ax (matplotlib.axes.Axes): Target subplot.
        coefficient (float): Linear slope or power-law prefactor.
        exponent (float): Power-law exponent; unused for linear fits.
        r_metric (float): Pearson r for linear; R^2 for power law.
        std_err (float or tuple): Error returned by fit_and_plot_model.
        model_type (str): 'linear' or 'exponential'.

    Returns:
        None: Subplot labels and annotations updated.

    Notes:
        Input units: q in angstrom^-1; rates in s^-1.
        Linear slope: angstrom^2/s; converted using 1e-8 to um^2/s.
        The displayed linear uncertainty is the fit standard error.
        Power-law prefactor: angstrom^n/s.
        Display the linear slope as D_app and the power-law prefactor as K.
        Nonfinite coefficients receive axis labels without numeric annotations.
    """
    # Set labels for the independent variable used in the fit.
    if model_type == 'linear':
        xlabel = r"$q^2$ ($\mathrm{\AA}^{-2}$)"
        title = r"Linear fit: $C = D_{\mathrm{app}}q^2$"
    elif model_type == 'exponential':
        xlabel = r"$q$ ($\mathrm{\AA}^{-1}$)"
        title = r"Power-law fit: $C = Kq^n$"
    else:
        raise ValueError(f"Unsupported rate-fit type: {model_type!r}")

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(r"Relaxation rate ($\mathrm{s}^{-1}$)", fontsize=14)
    ax.set_title(title, fontsize=16)

    # The fitting function already displays a message when fitting is unavailable.
    if not np.isfinite(coefficient):
        return

    if model_type == 'linear':
        # Convert both the slope and its fit standard error.
        converted_slope = coefficient * 1e-8
        converted_error = std_err * 1e-8

        slope_annotation = (
            r"Slope $D_{\mathrm{app}}$ ($\mathrm{\AA}^2/\mathrm{s}$) = "
            f"{coefficient:.3e}")
        converted_annotation = (
            r"$D_{\mathrm{app}}$ ($\mu\mathrm{m}^2/\mathrm{s}$)" "\n"
            f"{converted_slope:.4g} ({converted_error:.4g})")

        r_annotation = f"Pearson r = {r_metric:.4f}"

        # Place the original slope and Pearson r in the lower-right corner.
        # Place the converted slope and its uncertainty in the upper-left corner.
        annotations = [
            (slope_annotation, 0.98, 0.05, 'right', 'bottom'),
            (r_annotation, 0.98, 0.10, 'right', 'bottom'),
            (converted_annotation, 0.02, 0.95, 'left', 'top')
        ]

    else:
        if not np.isfinite(exponent):
            return

        # Annotate the power-law prefactor, exponent and R^2.
        prefactor_annotation = (
            r"Prefactor $K$ ($\mathrm{\AA}^{n}/\mathrm{s}$) = "
            f"{coefficient:.3e}")
        n_annotation = f"Exponent (n) = {exponent:.2f}"
        r2_annotation = fr"$R^2$ = {r_metric:.4f}"

        # Place all power-law annotations in the upper-left corner.
        annotations = [
            (prefactor_annotation, 0.02, 0.95, 'left', 'top'),
            (n_annotation, 0.02, 0.90, 'left', 'top'),
            (r2_annotation, 0.02, 0.85, 'left', 'top')
        ]

    # Positions are fractions of the axes width and height.
    for text, x, y, ha, va in annotations:
        ax.annotate(
            text, (x, y), xycoords='axes fraction',
            color="black", fontsize=12, ha=ha, va=va
        )

# Function to plot a specified parameter
def plot_params(q_values, y_values, ax, cmap, param_type, colors=None):
    """
    Plot gamma or PDI versus q.

    Parameters:
        q_values (array-like): Scattering vectors; one-dimensional.
        y_values (array-like): Corresponding parameter values.
        ax (matplotlib.axes.Axes): Target subplot.
        cmap (matplotlib.colors.Colormap): Colors indexed by input position.
        param_type (str): 'gamma' or 'PDI'.
        colors (list or None, optional): Explicit colors. Default: None.

    Returns:
        None

    Notes:
        Ignore nonfinite values and nonpositive q values.
        Preserve input color positions when filtering.
        Explicit colors take precedence over cmap.
        Require one color per input q value when colors is provided.
    """
    if param_type.lower() not in ('gamma', 'pdi'):
        raise ValueError("param_type must be 'gamma' or 'PDI'.")

    q_values = np.asarray(q_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)

    # Check alignment before filtering.
    if q_values.ndim != 1 or y_values.ndim != 1:
        raise ValueError("q_values and y_values must be one-dimensional.")
    if len(q_values) != len(y_values):
        raise ValueError("q_values and y_values must have equal lengths.")

    if colors is not None:
        colors = list(colors)
        if len(colors) != len(q_values):
            raise ValueError("colors must contain one entry per q value.")

    valid_mask = (
        np.isfinite(q_values)
        & np.isfinite(y_values)
        & (q_values > 0))
    valid_indices = np.flatnonzero(valid_mask)

    if valid_indices.size == 0:
        ax.text(0.5, 0.5, f"No valid {param_type} values",
                transform=ax.transAxes,
                ha='center', va='center', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return

    # Plot valid points using their original input colors.
    for i in valid_indices:
        color = cmap(int(i)) if colors is None else colors[i]
        ax.plot(q_values[i], y_values[i], 'o', color=color, markersize=10)

    # Set y-axis limits with a margin relative to the plotted values.
    y_min = np.min(y_values[valid_mask])
    y_max = np.max(y_values[valid_mask])
    value_scale = max(abs(y_min), abs(y_max))

    # Use 10% of the range, with a minimum of 1% of the value scale.
    y_margin = max(0.1 * (y_max - y_min), 0.01 * value_scale)

    # Keep a visible range when all plotted values are zero.
    if y_margin == 0:
        y_margin = 0.01

    ax.set_ylim(y_min - y_margin, y_max + y_margin)

# Function to configure a subplot with title, labels, and a combined legend
def configure_subplot(ax, title, xlabel, ylabel, exp_lines, exp_labels,
                      fit_lines, fit_labels):
    """
    Set subplot labels and a combined q / R^2 legend.

    Parameters:
        ax (matplotlib.axes.Axes): Target subplot.
        title (str): Subplot title.
        xlabel (str): Horizontal-axis label.
        ylabel (str): Vertical-axis label.
        exp_lines (list): Experimental-data line objects.
        exp_labels (list): Corresponding q labels.
        fit_lines (list): Fitted-curve line objects.
        fit_labels (list): Corresponding R^2 labels.

    Returns:
        None

    Notes:
        Require matching list lengths and ordering.
        Use experimental markers as legend handles.
    """
    # Check that each q has one experimental line, fitted line, and label pair.
    if not (len(exp_lines) == len(exp_labels)
            == len(fit_lines) == len(fit_labels)):
        raise ValueError(
            "Experimental/fit line and label lists must have equal lengths.")

    # Set the title and labels for the subplot.
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel.replace("Beta", "β"), fontsize=14)

    # Combine q values and R^2 values in a single legend entry.
    combined_labels = [f"{exp_label} - {fit_label}"
        for exp_label, fit_label in zip(exp_labels, fit_labels)]

    # Keep one legend and let Matplotlib choose its position.
    if combined_labels:
        ax.legend(
            exp_lines,
            combined_labels,
            loc='best',
            fontsize=8,
            frameon=True,
            framealpha=0.85,
            borderaxespad=0.5)
    else:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

# Function to add text at the least obstructive position
def add_text_best_position(ax, text, fontsize=12):
    """
    Add text near the least obstructed candidate position.

    Parameters:
        ax (matplotlib.axes.Axes): Target subplot.
        text (str or None): Annotation text.
        fontsize (int or float, optional): Font size. Default: 12.

    Returns:
        matplotlib.text.Text or None: Added text; None for empty input.

    Notes:
        Test eight positions using approximate text-box dimensions.
        Penalize overlap with visible line samples and the current legend.
        Placement is approximate; inspect long annotations and dense plots.
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

        # Update limits and layout before transforming plotted data.
    ax.figure.canvas.draw()

    # Collect visible plotted data in axes coordinates.
    points_axes = []
    for line in ax.lines:
        if not line.get_visible():
            continue

        x_data = np.asarray(line.get_xdata(orig=False), dtype=float)
        y_data = np.asarray(line.get_ydata(orig=False), dtype=float)

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
            renderer = ax.figure.canvas.get_renderer()
            legend_bbox_display = legend.get_window_extent(
                renderer=renderer
            )
            legend_bbox_axes = ax.transAxes.inverted().transform(
                legend_bbox_display.get_points()
            )
            legend_box = (
                legend_bbox_axes[0, 0],
                legend_bbox_axes[1, 0],
                legend_bbox_axes[0, 1],
                legend_bbox_axes[1, 1]
            )
        except (AttributeError, RuntimeError, ValueError):
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
    
