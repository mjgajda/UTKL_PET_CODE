import numpy as np
from ..GRAPH_CLASSES.Histogram import * # Keeping this class as is

# time class for analyzing the time attributes of PET datasets
class CTR:
    def __init__(self, coincObj):
        """
        Initialize CTR with the coincObj and calculate the time-of-flight (TOF) values.
        """
        print('Defining CTR')
        self.coincObj = coincObj

        print('Calculating TOF')
        self.tof = np.subtract(self.coincObj.channel_1_time, self.coincObj.channel_2_time)

        print("Creating a histogram")
        self.CTR = Histogram(self.tof)

    def isolate_CTR(self, numberSigma):
        """
        Isolate the CTR (time-of-flight) distribution by fitting a Gaussian and applying
        a filter based on a sigma threshold.
        """
        try:
            # Step 1: Fit the Gaussian to the TOF data
            print('Fitting Gauss')
            self.CTR.get_Gauss_Variables()
            A, mu, sigma = self.CTR.popt  # Assuming popt is (A, mu, sigma)

            # Assuming self.coincObj.data is the array of data points
            lower_bound = mu - numberSigma * sigma
            upper_bound = mu + numberSigma * sigma
            
            # Create the mask for values between [mu - numberSigma*sigma, mu + numberSigma*sigma]
            mask = (self.tof >= lower_bound) & (self.tof <= upper_bound)
            
            # Step 3: Apply the mask to filter out the data
            print('Applying mask...')
            self.coincObj.apply_combined_mask(mask)



            return True

        except Exception as e:
            print(f"Error during CTR isolation: {e}")
            return False

    def timeCut(self, numberSigma, threshold):
        """
        Apply time cuts to the dataset based on the isolated CTR and a specified sigma threshold.
        """
        if not self.isolate_CTR(numberSigma):
            return None

        # Assuming threshold validation is needed here
        # # For example, if the number of filtered events is below the threshold, return None
        # if np.sum(self.tof) < threshold:
        #     print("Time cut threshold not met")
        #     return None
        
        return self.coincObj
