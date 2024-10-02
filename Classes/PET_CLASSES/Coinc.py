import concurrent.futures
import numpy as np
from numpy import ma
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import concurrent
from ..GRAPH_CLASSES.Histogram import *
from PET_Data import PETData
import numpy as np
from numpy import ma
from concurrent.futures import ThreadPoolExecutor

class Coinc(PETData):

    def create_combined_mask(self, condition_func_1, condition_func_2):
        """
        Create a combined mask for both channel 1 and channel 2 based on provided conditions
        using ThreadPoolExecutor for parallelization.

        Parameters:
        - condition_func_1: A function that generates a boolean mask for channel 1.
        - condition_func_2: A function that generates a boolean mask for channel 2.

        Returns:
        - A combined boolean mask for both channels.
        """
        with ThreadPoolExecutor() as executor:
            # Execute the mask creation in parallel for both channels
            future_mask_channel_1 = executor.submit(condition_func_1, self.channel_1_energy)
            future_mask_channel_2 = executor.submit(condition_func_2, self.channel_2_energy)

            # Wait for both tasks to complete
            mask_channel_1 = future_mask_channel_1.result()
            mask_channel_2 = future_mask_channel_2.result()

        # Combine the masks using a logical AND operation
        combined_mask = mask_channel_1 & mask_channel_2
        return combined_mask

    def apply_combined_mask(self, combined_mask):
        """
        Apply the combined mask to filter data from both channel 1 and channel 2 arrays
        using ThreadPoolExecutor for parallelization.

        Parameters:
        - combined_mask: A boolean mask to apply to both channel 1 and channel 2 data.
        """
        if not isinstance(combined_mask, np.ndarray):
            raise ValueError("Mask must be a NumPy array")

        with ThreadPoolExecutor() as executor:
            future_channel_1 = executor.submit(self._apply_mask_to_channel_1, combined_mask)
            future_channel_2 = executor.submit(self._apply_mask_to_channel_2, combined_mask)

            # Wait for both tasks to complete
            future_channel_1.result()
            future_channel_2.result()

    def _apply_mask_to_channel_1(self, combined_mask):
        """
        Helper function to apply the combined mask to channel 1 arrays using numpy boolean indexing.
        """
        self.channel_1_energy = self.channel_1_energy[combined_mask]
        self.channel_1_time = self.channel_1_time[combined_mask]
        self.channel_1_id = self.channel_1_id[combined_mask]

    def _apply_mask_to_channel_2(self, combined_mask):
        """
        Helper function to apply the combined mask to channel 2 arrays using numpy boolean indexing.
        """
        self.channel_2_energy = self.channel_2_energy[combined_mask]
        self.channel_2_time = self.channel_2_time[combined_mask]
        self.channel_2_id = self.channel_2_id[combined_mask]
    
    def _filter_by_channel_pair(self):
        """
        Internal method to filter the data to include only rows where a given 
        (channel_1_id, channel_2_id) pair is present.
        """
        # Create a boolean mask for the channel 1 and channel 2 ID comparisons
        print('making masks')
        mask_channel_1 = self.channel_1_id == self.channel_1_id_val
        mask_channel_2 = self.channel_2_id== self.channel_2_id_val

        # Combine the two masks using a logical AND operation
        print('combining masks')
        combined_mask = mask_channel_1 & mask_channel_2

        # Apply combined mask to filter data
        self.apply_combined_mask(combined_mask)

        return self

    def __init__(self, petDataObj, channel_1_id_val, channel_2_id_val):
        """
        Initialize the coinc class with an existing PETData object and specific channel ID pair to filter on.

        Parameters:
        - petDataObj: A PETData object that contains the raw data.
        - channel_1_id_val: The unique channel_1_id value to filter.
        - channel_2_id_val: The unique channel_2_id value to filter.
        """
        print('defining')
        self.channel_1_energy = petDataObj.channel_1_energy
        self.channel_1_time = petDataObj.channel_1_time
        self.channel_1_id = petDataObj.channel_1_id
        self.channel_2_energy = petDataObj.channel_2_energy
        self.channel_2_time = petDataObj.channel_2_time
        self.channel_2_id = petDataObj.channel_2_id

        self.channel_1_id_val = channel_1_id_val
        self.channel_2_id_val = channel_2_id_val

        # Filter the data based on the unique pair of channel IDs using the method
        print('filter')
        self._filter_by_channel_pair()
        print('end')
        print(self.channel_1_energy)
    

    def get_filtered_channel_data(self, channel):
        """
        Returns the filtered data for the specified channel (1 or 2).
        """
        if channel == 1:
            return self.channel_1_time, self.channel_1_energy, self.channel_1_id
        elif channel == 2:
            return self.channel_2_time, self.channel_2_energy, self.channel_2_id
        else:
            raise ValueError("Invalid channel number. Choose 1 or 2.")

# I/O-bound task (such as summing or filtering channels)
def sum_coinc_data(coinc_list):
    """
    Combine filtered data from multiple coinc objects and return a new PETData object
    with the combined data. This function is optimized for parallel execution using
    concurrent futures.

    Parameters:
    - coinc_list: List of coinc objects to be combined.

    Returns:
    - A list with the combined filtered data from all coinc objects.
    """
    def collect_data(attr):
        return [getattr(coinc_obj, attr) for coinc_obj in coinc_list]

    # Thread-based parallelism for I/O-bound tasks like summing or concatenating arrays
    with ThreadPoolExecutor() as executor:
        future_channel_1_time = executor.submit(np.concatenate, collect_data('channel_1_time'))
        future_channel_1_energy = executor.submit(np.concatenate, collect_data('channel_1_energy'))
        future_channel_1_id = executor.submit(np.concatenate, collect_data('channel_1_id'))

        future_channel_2_time = executor.submit(np.concatenate, collect_data('channel_2_time'))
        future_channel_2_energy = executor.submit(np.concatenate, collect_data('channel_2_energy'))
        future_channel_2_id = executor.submit(np.concatenate, collect_data('channel_2_id'))

        # Retrieve results
        combined_channel_1_time = future_channel_1_time.result()
        combined_channel_1_energy = future_channel_1_energy.result()
        combined_channel_1_id = future_channel_1_id.result()

        combined_channel_2_time = future_channel_2_time.result()
        combined_channel_2_energy = future_channel_2_energy.result()
        combined_channel_2_id = future_channel_2_id.result()

    # Return a list with combined arrays (PETData can be created with this data)
    return [
        combined_channel_1_time,
        combined_channel_1_energy,
        combined_channel_1_id,
        combined_channel_2_time,
        combined_channel_2_energy,
        combined_channel_2_id
    ]
