import numpy as np
import dask.array as da
import dask
from dask import delayed
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from scipy.optimize import curve_fit

class ScatterPlot:
    def __init__(self, data, ax=None, rc_params_path=None):
        """
        Initialize the ScatterPlot class.
        :param data: A list of lists. Each element is an array containing:
                     - x-values (array-like)
                     - y-values (array-like)
                     - (optional) z-values (array-like) for color.
        :param ax: Optional matplotlib axis object. If provided, plots will be drawn on this axis.
        :param rc_params_path: Optional path to a matplotlib rcParams file for plot customization.
        """
        # Store the axis and rc_params_path if provided
        self.ax = ax if ax is not None else plt.gca()
        self.rc_params_path = rc_params_path
        
        # If rc_params_path is provided, load the rcParams file
        if rc_params_path:
            rc_params_file = Path(rc_params_path)
            if rc_params_file.exists():
                mpl.rc_file(rc_params_file)
            else:
                raise FileNotFoundError(f"RC params file {rc_params_path} not found.")
        
        # Preprocess and store x, y, z data as Dask arrays
        self.x_data, self.y_data, self.z_data = self.process_data(data)
        
        # Initialize a list to store fit parameters for each dataset
        self.fit_params = []

    def process_data(self, data):
        """
        Process the input data and convert to Dask arrays.
        :param data: A list of lists. Each element contains x, y, and optionally z arrays.
        :return: Tuple of processed x, y, and z arrays (as Dask arrays).
        """
        x_data = []
        y_data = []
        z_data = []
        
        for plot in data:
            x = da.array(plot[0])  # Convert x to a Dask array
            y = da.array(plot[1])  # Convert y to a Dask array
            z = da.array(plot[2]) if len(plot) > 2 else None  # Convert z to Dask array if it exists

            x_data.append(x)
            y_data.append(y)
            z_data.append(z)

        return x_data, y_data, z_data

    @delayed
    def compute_scatter(self, x, y, z):
        """
        Delayed function to compute the scatter plot data points for a single dataset.
        This will be parallelized using Dask's delayed execution.
        """
        x_np = x.compute()  # Compute Dask array to NumPy
        y_np = y.compute()  # Compute Dask array to NumPy
        z_np = z.compute() if z is not None else None  # Compute Dask array to NumPy, if z exists
        
        return x_np, y_np, z_np

    @delayed
    def plot_scatter(self, x_np, y_np, z_np):
        """
        Delayed function to plot a scatter plot for the given x, y, and z data.
        """
        if z_np is not None:
            scatter = self.ax.scatter(x_np, y_np, c=z_np, cmap='viridis')
            self.ax.figure.colorbar(scatter, ax=self.ax, label='Color Bar')
        else:
            self.ax.scatter(x_np, y_np)

        return None  # Return None since we only want the plotting done

    @delayed
    def compute_fit_params(self, func, x, y):
        """
        Delayed function to compute fit parameters for a dataset.
        :param func: The function to fit the data.
        :param x: The x-data (NumPy array).
        :param y: The y-data (NumPy array).
        :return: Fit parameters.
        """
        params, _ = curve_fit(func, x, y)
        return params

    @delayed
    def plot_fitted_curve(self, func, x_np, params, label):
        """
        Delayed function to plot the fitted curve.
        :param func: The function used to fit the data.
        :param x_np: The x-values (NumPy array).
        :param params: The fit parameters for the function.
        :param label: Label for the plot.
        """
        y_fit = func(x_np, *params)
        self.ax.plot(x_np, y_fit, label=label, linestyle='--')

    def fit_data(self, functions):
        """
        Fits each dataset (x, y) to the provided functions and stores the fit parameters.
        :param functions: A list of functions, one for each dataset.
        """
        if len(functions) != len(self.x_data):
            raise ValueError("The number of functions must match the number of datasets.")
        
        self.fit_params = []  # Reset the fit_params list
        delayed_tasks = []

        # Perform curve fitting for each dataset in parallel
        for i in range(len(self.x_data)):
            x_np = self.x_data[i].compute()
            y_np = self.y_data[i].compute()
            func = functions[i]
            
            # Delay the curve fitting task
            delayed_task = self.compute_fit_params(func, x_np, y_np)
            delayed_tasks.append(delayed_task)

        # Execute the fitting in parallel
        self.fit_params = dask.compute(*delayed_tasks)

    def plot_fit(self, functions):
        """
        Plots the fitted curves using the stored fit parameters.
        :param functions: A list of functions used for fitting, one for each dataset.
        """
        if not self.fit_params:
            raise ValueError("Fit parameters not found. Run `fit_data` first.")
        
        # Collect delayed plotting tasks
        delayed_tasks = []
        
        # Plot the fitted curves in parallel
        for i in range(len(self.x_data)):
            x_np = self.x_data[i].compute()
            func = functions[i]
            params = self.fit_params[i]
            
            # Delay the plotting task
            delayed_task = self.plot_fitted_curve(func, x_np, params, label=f"Fitted Curve {i+1}")
            delayed_tasks.append(delayed_task)

        # Execute the plotting in parallel
        dask.compute(*delayed_tasks)

        # Finalize and show the plot
        self.ax.legend()

    def plot(self):
        """
        Generate the scatter plot using the precomputed data.
        This function computes the Dask arrays lazily in parallel and plots them.
        """
        fig = plt.figure(figsize=(8, 6))
        
        # Collect delayed plotting tasks for each dataset
        delayed_tasks = []
        for i in range(len(self.x_data)):
            # Compute x_np, y_np, z_np using delayed computation
            x_np, y_np, z_np = self.compute_scatter(self.x_data[i], self.y_data[i], self.z_data[i])
            
            # Create a delayed task for plotting the scatter plot
            delayed_tasks.append(self.plot_scatter(x_np, y_np, z_np))
        
        # Trigger the delayed computations and plotting in parallel
        dask.compute(*delayed_tasks)
        
        # Show the plot
