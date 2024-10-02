import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Define the Gaussian function for curve fitting
def gaussian(x, amplitude, mean, std):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * std ** 2))


class Histogram:
    
    def compute_histogram(self, data):
        """Compute the histogram for the data."""
        print('plotting data')
        # Find the maximum value
        max_value = np.max(data)

        # Find the minimum value
        min_value = np.min(data)

        # Calculate the difference
        difference = abs(max_value - min_value)
        print(data)

        # if the difference between the minimum and maximum value is calculated as 0 
        # use 20 bins
        if int(difference * 10) == 0:

            histData, bins = np.histogram(data, bins=20)

        # If the diffence is not zero then use 10 times the difference for the number
        # then use 10 times the difference for the number of bins
        else:
            histData, bins = np.histogram(data, bins=int(difference * 50))
        print('defining x cordinates')

        # find the center of each bin by taking the average of the edges
        binCenters = 0.5 * (bins[1:] + bins[:-1])
        print('getting standard deviatoin')
        print(histData)

        # calculate the standard deviation of the data
        std = np.std(data)
        return histData, binCenters, std

    def __init__(self, numpyData, settings=None):
        print('settings')
        self.settings = settings

        # Sequentially compute histograms for each dataset
        print('making hist')
        results = self.compute_histogram(numpyData)


        print('defining results')
        self.histData, self.binCenters, self.std = results
        self.maxHeight = None
        self.mean = None
        self.popt = None
        self.smoothedCounts = None
        self.peaks = None
        self.peakProperties = None

    def identify_max_height_bin(self):
        """Identify the bin with the maximum height for each dataset."""

        self.maxHeight=(self.histData.max())
        self.mean = (self.binCenters[self.histData.argmax()])

    def fit_gaussian(self):
        """Fit a Gaussian to the histogram data."""
        try:
            popt, _ = curve_fit(
                gaussian,
                self.binCenters,
                self.histData,
                p0=[self.maxHeight, self.mean, self.std]
            )
            return popt
        except RuntimeError:
            return None

    def get_gauss_variables(self):
        """Fit Gaussians to all datasets sequentially."""
        if self.mean is None:
            self.identify_max_height_bin()

        # Sequential Gaussian fitting for each dataset
        self.popt = self.fit_gaussian() 

    def plot_single_gaussian(self, ax):
        """Plot a single Gaussian on the axis."""
        if self.popt is None:
            print(f"Gaussian fit failed for dataset")
            return

        amplitudeFitted, meanFitted, stddevFitted = self.popt
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, int((xmax - xmin) * 10))
        y = gaussian(x, amplitudeFitted, meanFitted, stddevFitted)

    def plot_gauss(self):
        """Plot all Gaussians sequentially."""
        if self.popt is None:
            self.get_gauss_variables()

        self.plot_single_gaussian()

        # self.ax.legend()
    
    # Method to plot histogram on a given axis object
    def plot_histogram(self, ax):
        """Plot the histogram on the provided matplotlib axis."""
        ax.bar(self.binCenters, self.histData, width=(self.binCenters[1] - self.binCenters[0]))
    
    def smooth_histogram(self,sigma=None):
        #intialize variables if none are specified
        if sigma is None:
            sigma = self.std

        self.smoothedCounts = gaussian_filter1d(self.histData, sigma=sigma)

    
    def find_peaks(self,sigma=None, height = None):

        if height is None:
            height = max(self.histData) / 10

        # if the smooth counts are not calculated calculate them
        if self.smoothedCounts is None:
            self.smooth_histogram()

        self.peaks, self.peakProperties = find_peaks(self.smoothedCounts, height )
    

    
 
    


    def _zero_below_threshold_single(self,  threshold_energy):
        """Helper function to zero out histogram data below the threshold for a single dataset."""
        self.histData[self.binCenters < threshold_energy] = 0
