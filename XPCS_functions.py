import tkinter as tk
from tkinter import Canvas, messagebox, ttk
from PyQt5.QtWidgets import QApplication, QFileDialog, QWidget, QVBoxLayout, QPushButton, QDesktopWidget
import os
import h5py
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import warnings
from scipy.stats import pearsonr

#-----------------------------------------------------------------------#
#--------------------- Directory and File Handling ---------------------#
#-----------------------------------------------------------------------#

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
    window_height = 200
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

    sirius_new_button = tk.Button(root, text="Sirius new", command=set_sirius_new)
    sirius_new_button.pack(pady=5)
    
    APS_button = tk.Button(root, text="APS", command=set_APS)
    APS_button.pack(pady=5)

    ESRF_button = tk.Button(root, text="ESRF", command=set_ESRF)
    ESRF_button.pack(pady=5)

    # Bind the closing of the window to the on_closing function
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start the main GUI event loop
    root.mainloop()
    
    return synchrotron

# Function to select a directory or files
def select_directory_or_files():
    """
    Displays a GUI window to select a directory or individual files and returns the selected directory path and list of files.

    Returns:
        str: The selected directory path.
        list: List of selected files.
    """
    app = QApplication([])

    layout = QVBoxLayout()

    directory_button = QPushButton("Select Directory")
    files_button = QPushButton("Select Files")

    directory_path = ""
    selected_files = []

    # Function to select a directory
    def select_directory():
        nonlocal directory_path
        directory_path = QFileDialog.getExistingDirectory(None, "Select Directory")
        # Close the window after the user has made a selection
        window.close()

    # Function to select specific files
    def select_files():
        nonlocal selected_files, directory_path
        # Get the selected file paths
        selected_file_paths, _ = QFileDialog.getOpenFileNames(None, "Select Files", "", "HDF5 Files (*.hdf5);;All Files (*)")

        if not selected_file_paths:
            # If no files were selected, set the directory path to an empty string
            directory_path = ""
        else:
            # Set the directory path to the common directory of the selected files
            directory_path = os.path.dirname(selected_file_paths[0])

            # Extract only the filenames (without the path) from the selected file paths
            selected_files.extend([os.path.basename(file_path) for file_path in selected_file_paths])

        # Close the window after the user has made a selection
        window.close()

    directory_button.clicked.connect(select_directory)
    files_button.clicked.connect(select_files)

    layout.addWidget(directory_button)
    layout.addWidget(files_button)

    window = QWidget()
    window.setLayout(layout)

    # Set the window size and position it in the center of the screen
    window_width = 300
    window_height = 100
    screen_geometry = QDesktopWidget().screenGeometry()
    x_position = (screen_geometry.width() - window_width) // 2
    y_position = (screen_geometry.height() - window_height) // 2

    window.setGeometry(x_position, y_position, window_width, window_height)

    # Set the window title
    window.setWindowTitle("Select Directory or Files")
    window.show()

    app.exec_()

    if directory_path and not selected_files:
        # If a directory was selected but no files were chosen, get the list of .hdf5 files in that directory
        selected_files = [os.path.basename(file) for file in os.listdir(directory_path) if file.endswith('.hdf5')]

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

    # Ask the user if they want to define the range
    user_response = messagebox.askyesno("Define Range", "Do you want to define the range of 't'?")

    return user_response

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
    elif selected_synchrotron == 'ESRF':
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

# Function to get de t range
def get_t_range_slider(t_values):
    """
    Display a centered GUI window for the user to select a range for 't' using sliders.

    Args:
        t_values (list): List of 't' values.

    Returns:
        tuple: Lower and upper limits of the selected range.
    """
    def apply_range():
        nonlocal lower_limit, upper_limit
        lower_limit = int(lower_slider.get())
        upper_limit = int(upper_slider.get())

        if lower_limit > upper_limit:
            messagebox.showerror("Error", "Lower limit must be less than or equal to the upper limit.")
        else:
            range_dialog.destroy()

    def on_arrow_key_press(event):
        """
        Move the slider position by one when arrow keys are pressed.
        """
        if event.keysym == 'Left':
            lower_slider.set(lower_slider.get() - 1)
        elif event.keysym == 'Right':
            lower_slider.set(lower_slider.get() + 1)
        elif event.keysym == 'Up':
            upper_slider.set(upper_slider.get() + 1)
        elif event.keysym == 'Down':
            upper_slider.set(upper_slider.get() - 1)

    range_dialog = tk.Toplevel()
    range_dialog.title("Select 't' Range")

    # Center the window on the screen
    window_width = 400
    window_height = 180
    screen_width = range_dialog.winfo_screenwidth()
    screen_height = range_dialog.winfo_screenheight()
    x_position = (screen_width - window_width) // 2
    y_position = (screen_height - window_height) // 2

    range_dialog.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

    lower_limit = 0
    upper_limit = len(t_values) - 1

    # Create sliders for lower and upper limits with 't' values as labels
    lower_slider = tk.Scale(range_dialog, from_=0, to=len(t_values) - 1, orient=tk.HORIZONTAL, label="◄ Lower Limit ►",
                            length=300, showvalue=0, command=lambda x: lower_slider_label.config(text=f"{t_values[int(x)]:.4f}"))
    lower_slider.set(lower_limit)
    lower_slider.pack()

    lower_slider_label = tk.Label(range_dialog, text=f"{t_values[lower_limit]:.4f}", width=10)
    lower_slider_label.pack()

    upper_slider = tk.Scale(range_dialog, from_=0, to=len(t_values) - 1, orient=tk.HORIZONTAL, label="▲ Upper Limit ▼",
                            length=300, showvalue=0, command=lambda x: upper_slider_label.config(text=f"{t_values[int(x)]:.4f}"))
    upper_slider.set(upper_limit)
    upper_slider.pack()

    upper_slider_label = tk.Label(range_dialog, text=f"{t_values[upper_limit]:.4f}", width=10)
    upper_slider_label.pack()

    # Create a button to apply the selected range
    apply_button = tk.Button(range_dialog, text="Apply Range", command=apply_range)
    apply_button.pack()

    # Bind arrow key events to move the sliders
    range_dialog.bind('<Left>', on_arrow_key_press)
    range_dialog.bind('<Right>', on_arrow_key_press)
    range_dialog.bind('<Up>', on_arrow_key_press)
    range_dialog.bind('<Down>', on_arrow_key_press)

    # Wait for the window to be closed
    range_dialog.wait_window()

    return lower_limit, upper_limit

# Function to obtain 't' range limits
def get_t_range_limits(dataset_df, define_t_range):
    """
    Get the lower and upper limits for 't'.

    Args:
        dataset_df (DataFrame): The dataset DataFrame.
        define_t_range (bool): Flag indicating if 't' range is defined.

    Returns:
        tuple: Lower and upper limits for 't'.
    """
    # Extract 't' values from the specified DataFrame column
    t_values = dataset_df['# t'].tolist()

    if define_t_range:
        # Get the range using a slider
        return get_t_range_slider(t_values)
    else:
        # Use the full range of 't' values
        return 0, len(t_values) - 1

#-----------------------------------------------------------------------#
#--------------------------- Initializations ---------------------------#
#-----------------------------------------------------------------------#

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
        "q_value": [],
        "q_square": [],   
        "relax_rate_single": [],
        "relax_rate_stretched": [],
        "gamma": [],  
        "relax_rate_cumulants": [],
        "PDI": []
    }
        
    return derived_params

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

    def create_graph(self):
        """Create the interactive graph."""
        self.figure = Figure(figsize=(8, 6))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
        
        # Iterate over the columns and adjust the code
        for i, q_column in enumerate(self.new_dataset_df.columns[1:], 1):
            
            # Extract the q value from the column name
            match = re.search(r"q=(\d+\.\d+)", q_column)
            if match:
                q_value = float(match.group(1))
            
            # Get the 't' and 'g2' data for the current q value
            t_q = self.new_dataset_df['# t']
            g2_q = self.new_dataset_df[q_column]

            # Slice the values of 't' and 'g2' using the provided indices
            t_q = t_q.iloc[self.lower_limit:self.upper_limit + 1]
            g2_q = g2_q.iloc[self.lower_limit:self.upper_limit + 1]

            # Fit the Single Exponential Model
            fit_params_selq, fitted_curve_selq, r2_selq = fit_single_exponential(t_q, g2_q)
                
            A_selq = fit_params_selq[0]
            B_selq = fit_params_selq[1]

            ### Plot the experimental data and the fitted curves ###

            color = self.cmap(i-1)  # Get color based on index
                
            # Plot experimental data points
            line_exp_sq = self.ax.semilogx(t_q, (g2_q - A_selq)/B_selq, 'o', color=color)[0]

            # Plot fitted curve
            line_fit_sq = self.ax.semilogx(t_q, (fitted_curve_selq - A_selq)/B_selq, color=color, linestyle='-')[0]

            # Add labels to the legend lists based on the model type
            self.lines_labels_dict["exp_lines_selq"].append(line_exp_sq)
            self.lines_labels_dict["exp_labels_selq"].append(f"q = {q_value:.6f}")
            self.lines_labels_dict["fit_lines_selq"].append(line_fit_sq)
            self.lines_labels_dict["fit_labels_selq"].append(f"R2: {r2_selq:.3f}")
                
        ### Configure the plot ###
        
        # Configure legend and title
        self.ax.set_title('Correlation function fitting - Single Exponential Model')
        self.ax.set_xlabel('Delay Time (s)')
        self.ax.set_ylabel(r'$(g_2 - \mathrm{baseline}) / \beta$')
        
        # Add labels to the plot
        # Combine q and R values
        combined_labels = [f"{exp_label} - {fit_label}" for exp_label, fit_label in
                           zip(self.lines_labels_dict["exp_labels_selq"], self.lines_labels_dict["fit_labels_selq"])]
        
        # Create the legend for experimental data
        legend_exp = self.ax.legend(self.lines_labels_dict["exp_lines_selq"], combined_labels,
                                    loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0)
        self.ax.add_artist(legend_exp)

        self.canvas.draw()

    def create_q_selector(self):
        """Create the q value selector checkboxes."""
        for i, q_column in enumerate(self.new_dataset_df.columns[1:], 1):
            # Use regular expression to extract q value
            match = re.search(r"q=(\d+\.\d+)", q_column)
            if match:
                q_value = float(match.group(1))

                # Initially selected state
                #initial_state = tk.NORMAL

                # Create a custom style for the Checkbutton
                style = ttk.Style()
                style.configure("TCheckbutton", indicatorrelief="flat", relief="flat")

                btn_var = tk.BooleanVar(value=True)  # Use a separate BooleanVar
                btn = ttk.Checkbutton(self.root, text=f'q = {q_value:.6f}', style="TCheckbutton",
                                    variable=btn_var, #state=initial_state,
                                    command=lambda q=q_value, var=btn_var: self.toggle_curve(q, var))

                btn.invoke()  # Invoke the button to set its initial state

                btn.pack()

                # Append the button and its associated variable to the q_buttons list
                self.q_buttons.append((btn, btn_var))

    def toggle_curve(self, q_value, var):
        """Toggle visibility of the curve for the specified q value."""
        # Find the index of q in the q list
        q_index = self.lines_labels_dict["exp_labels_selq"].index(f"q = {q_value:.6f}")

        # Toggle visibility of the experimental curve and the fitted curve
        exp_line = self.lines_labels_dict["exp_lines_selq"][q_index]
        fit_line = self.lines_labels_dict["fit_lines_selq"][q_index]

        exp_line.set_visible(not exp_line.get_visible())
        fit_line.set_visible(not fit_line.get_visible())

        # Update the selected values list based on the visibility
        self.selected_values = [
            float(re.search(r"q=(\d+\.\d+)", column).group(1))
            for column, line in zip(self.new_dataset_df.columns[1:], self.lines_labels_dict["exp_lines_selq"])
            if line.get_visible()
        ]

        # Update the selected values list based on the visibility
        self.selected_columns = [
            q for q, line in zip(self.new_dataset_df.columns[1:], self.lines_labels_dict["exp_lines_selq"])
            if line.get_visible()
        ]

        # Redraw the canvas to reflect the changes
        self.canvas.draw()

    def create_buttons(self):
        """Create the Select and Exit buttons."""
        ttk.Button(self.root, text="Select", command=self.select_values).pack(side='left', padx=10)
        ttk.Button(self.root, text="Exit", command=self.exit_window).pack(side='right', padx=10)

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
        """Close the window without saving changes."""
        self.root.destroy()

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
    fit_params_single, fitted_curve_single, r2_single = fit_model_with_constraints(t, g2, single_exponential, initial_params_single)
    #fit_params_single0, fitted_curve_single0, r2_single0 = fit_model(t, g2, single_exponential, initial_params_single)

    return fit_params_single, fitted_curve_single, r2_single

# Function to fit a stretched exponential model to the provided data
def fit_stretched_exponential(t, g2, fit_params_single):
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
    fit_params_stretched, fitted_curve_stretched, r2_stretched = fit_model_with_constraints(t, g2, stretched_exponential, initial_params_stretched)
    #fit_params_stretched0, fitted_curve_stretched0, r2_stretched0 = fit_model(t, g2, stretched_exponential, initial_params_stretched)
        
    return fit_params_stretched, fitted_curve_stretched, r2_stretched

# Function to fit a cumulants model to the provided data
def fit_cumulants(t, g2, fit_params_single):
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
    fit_params_cumulants, fitted_curve_cumulants, r2_cumulants = fit_model_with_constraints(t, g2, cumulants_model, initial_params_cumulants)
    #fit_params_cumulants0, fitted_curve_cumulants0, r2_cumulants0 = fit_model(t, g2, cumulants_model, initial_params_cumulants)
        
    return fit_params_cumulants, fitted_curve_cumulants, r2_cumulants

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
        fitted_curve_stretched = A_stretched + B_stretched * np.exp(-2 * C_stretched * t) ** gamma
        fitted_curve_cumulants = A_cumulants + B_cumulants * np.exp(-2 * C1_cumulants * t) * (1 + (1 / 2) * C2_cumulants * t**2)**2

        for i in range(len(t)):
            row_str = f"{t[i]}\t{g2[i]}\t{np.std(g2)}\t{fitted_curve_single[i]}\t{fitted_curve_stretched[i]}\t{fitted_curve_cumulants[i]}\n"
            dat_file.write(row_str)

# Function to print error and success counters
def print_summary(counters):
    """
    Print a summary of processing results.

    Parameters:
        counters (dict): A dictionary containing counters and lists for error and success tracking.
    """
    print(f"Total HDF5 files: {counters['hdf5_files']}")
    print(f"Total q values: {counters['q_values']}")
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

# Define the function to fit and plot a specified model
def fit_and_plot_model(x_values, relax_rate_values, ax, cmap, label, model_type='linear'):
    """
    Fit a model to given data and generate a plot on a specified subplot

    Args:
        x_values (array-like): Array of q2 or q values.
        relax_rate_values (array-like): Array of corresponding relaxation rates.
        ax (matplotlib.axes.Axes): The subplot where the plot will be generated.
        cmap (matplotlib.colors.Colormap): Colormap for line colors.
        label (str): Label for the plot
        model_type (str, optional): Type of model to fit. Supported values are 'linear' or 'exponential'. Defaults to 'linear'.

    Returns:
        D (float): Diffusion coefficient obtained from the fit.
        n (float): Exponent obtained from the fit (applicable for exponential model).
        r_metric (float): Metric (e.g., Pearson correlation coefficient or R-squared) indicating the quality of the fit.
        std_err (float): Standard error of the slope for linear fits. Tuple (std_err_D, std_err_n) for exponential fits.
    """
    try:
        if model_type == 'linear':
            # Call the function to perform the linear fit
            D, r_metric, std_err = fit_linear_model(x_values, relax_rate_values)
            n = np.nan  # For consistency, set n to NaN for linear fits
        elif model_type == 'exponential':
            # Call the function to perform the exponential fit
            D, n, r_metric, std_err_D, std_err_n = fit_exponential_model(x_values, relax_rate_values)
            std_err = (std_err_D, std_err_n)  # Standard error for exponential fits
        else:
            raise ValueError("Invalid model_type. Supported values are 'linear' or 'exponential'.")

        # Create a color palette with the number of points
        n_points = len(x_values)

        # Plot the model data
        for i in range(n_points):
            color = cmap(i)
            
            line_data = ax.plot(x_values[i], relax_rate_values[i], 'o', color=color, markersize=10, label='')[0]

        # Define the exponential function for fitting
        def exponential_function(q, D, n):
            return D * q**n

        # Generate the fit line using the fitted values of D and n
        if model_type == 'linear':
            q_fit = np.linspace(0, max(x_values), 100)
            fit_line = D * q_fit
        elif model_type == 'exponential':
            #q_fit = np.linspace(min(x_values), max(x_values), 1000)
            q_fit = np.linspace(min(x_values), max(x_values), 100)
            fit_line = exponential_function(q_fit, D, n)
        else:
            fit_line = None

        # Plot the fit line
        #ax.plot(q_fit, fit_line, linestyle='-', color='black', label=f"{model_type.capitalize()} Fit")
        ax.plot(q_fit, fit_line, linestyle='-', color='black', label='')

        # Remove the legend if it exists
        legend = ax.get_legend()
        if legend:
            legend.remove()

        # Set the axes to start from zero
        ax.set_xlim(0, (max(x_values) + 0.1 * max(x_values)))
        ax.set_ylim(0, max(relax_rate_values) * 1.1)
        #ax.set_ylim(0, (max(relax_rate_values) + 0.1 * max(relax_rate_values)))

        # Disable legend not found warning
        warnings.resetwarnings()
        
        return D, n, r_metric, std_err

    except ValueError as e:
        print(f"Error fitting and plotting the {model_type} model: {e}")
        return np.nan, np.nan, np.nan, np.nan

# Define the function to plot a specified param
def plot_params(q_values, y_values, ax, cmap, param_type):
    """
    Plot points of a derived param on a specified subplot.

    Args:
        q_values (array-like): Array of q values.
        y_values (array-like): Array of param values (gamma or PDI).
        ax (matplotlib.axes.Axes): The subplot where the plot will be generated.
        cmap (matplotlib.colors.Colormap): Colormap for point colors.
        param_type (str): Type of parameter ('gamma' or 'PDI').

    Returns:
        None
    """
    try:
        # Create a color palette with the number of points
        n_points = len(q_values)

        # Plot the data points with labels
        for i in range(n_points):
            color = cmap(i)
            ax.plot(q_values[i], y_values[i], 'o', color=color, markersize=10)

        # Set y-axis limits with a margin
        if param_type.lower() == 'gamma':
            y_min = np.min(y_values) - 0.01
            y_max = np.max(y_values) + 0.01
            ax.set_ylim(y_min, y_max)
        
        # Set y-axis limits with a small margin
        elif param_type.lower() == 'pdi':
            y_margin = 0.1 * (max(y_values) - min(y_values))
            ax.set_ylim(min(y_values) - y_margin, max(y_values) + y_margin)

    except ValueError as e:
        print(f"Error plotting points: {e}")
        
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
    #legend_exp = ax.legend(exp_lines, exp_labels, loc='upper right', bbox_to_anchor=(0.86, 1), borderaxespad=0)
    #ax.add_artist(legend_exp)

    # Create the legend for fitted curves
    #legend_fit = ax.legend(fit_lines, fit_labels, loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0)
    #ax.add_artist(legend_fit)

    # Combine q and R values
    combined_labels = [f"{exp_label} - {fit_label}" for exp_label, fit_label in zip(exp_labels, fit_labels)]

    # Create the legend for experimental data
    legend_exp = ax.legend(exp_lines, combined_labels, loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0)
    ax.add_artist(legend_exp)

    #print("Combined Labels:", combined_labels)

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


    
    
