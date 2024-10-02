import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from ..GRAPH_CLASSES.Histogram import *

class Energy:
    def __init__(self, coincObj):
        """
        Initialize the energy class with coincObj and create histograms
        for channel_1_energy and channel_2_energy.
        """
        self.coincObj = coincObj
        print('HistL')
        print(coincObj.channel_1_energy)
        print(coincObj.channel_1_energy)
        self.histL = Histogram(coincObj.channel_1_energy)
        print('HistR')
        self.histR = Histogram(coincObj.channel_2_energy)
    
    def detect_photopeaks(self, histogram):
        """Smooth histogram and detect the peak."""
        try:
            # smoothign the data in the energy histogram given
            histogram.smooth_histogram()

            # finding the peaks of the endergy histogram
            histogram.find_peaks()


            if len(histogram.peaks) > 1:
                # Sort peaks by height in descending order
                sorted_peak_indices = np.argsort(histogram.peakProperties['peak_heights'])[::-1]

                # Select the top 20% of peaks
                top_20_percent_count = max(1, int(len(histogram.peaks) * 0.2))
                top_peaks = sorted_peak_indices[:top_20_percent_count]

                # Get the peak index with the maximum height from the top 20%
                peak_idx = histogram.peaks[top_peaks[0]]  # Since sorted by height, the first will be the highest
            else:
                # If only one peak is detected, we simply return it
                peak_idx = histogram.peaks[0]

            print('Selected peak index:', peak_idx)
            peak_energy = self.binCenters[peak_idx]
            return peak_energy, self.smoothed_counts, peak_idx

        except Exception as e:
            print('Error during peak detection:', e)
            return None
    
    def fit_photopeak(self, hist, peak_energy):
        """Fit Gaussian to the detected peak."""
        fit_region = (hist.binCenters > peak_energy - 5) & (hist.binCenters< peak_energy + 5)
        popt, _ = curve_fit(gaussian, hist.binCenters[fit_region], hist.smoothed_counts[fit_region], p0=[100, peak_energy, 5])
        A, mu, sigma = popt
        return A, mu, sigma


        

    def isolate_photopeak(self, numberSigma):
        """
        Isolate the photopeak for both histograms (left and right channels)
        by detecting peaks, fitting Gaussian, and applying filters based on sigma.
        """
        try:
            masks = []
            print('Photopeaks')
            for hist, side in [(self.histL, 'left'), (self.histR, 'right')]:
                # Step 1: Detect the peak
                print('Finding peak')
                peak_energy, smoothed_counts, peak_idx = self.detect_photopeaks(hist)

                # Step 2: Fit a Gaussian around the peak
                print('Fitting the photopeak')
                A, mu, sigma = self.fit_photopeak(hist, peak_energy)

                # Step 3: Create a mask based on the numberSigma threshold
                print('Creating filter')
                masks.append(lambda x: x >= (mu - numberSigma * sigma))


            # Combine masks for both histograms
            combined_mask = self.coincObj.create_combined_mask(masks[0], masks[1])

            # Step 5: Apply the combined mask to filter the data
            print('Applying filter')
            self.coincObj.apply_combined_mask(combined_mask)

            return True

        except Exception as e:
            print(f"Error during photopeak isolation: {e}")
            return False

    def cut_on_photopeak_events(self, threshold):
        """
        Ensure that the histogram data size for both channels is above the threshold.
        """
        try:
            for hist in [self.histL, self.histR]:
                if hist.histData.size < threshold:
                    raise ValueError("Histogram size is below the threshold")
            return True
        except Exception as e:
            print(f"Error during photopeak event cut: {e}")
            return False

    def energyCuts(self, numberSigma, threshold):
        """
        Perform energy cuts based on photopeak isolation and apply filtering.
        """
        if not self.isolate_photopeak(numberSigma):
            return None

        if not self.cut_on_photopeak_events(threshold):
            return None
        
        return self.coincObj