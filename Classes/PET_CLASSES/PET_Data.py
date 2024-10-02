import numpy as np

class PETData:
    def __init__(self, path):
        """
        Initialize PETData by loading the tab-separated file into NumPy arrays.
        """
        # Load the tab-separated file using numpy's loadtxt function
        data = np.loadtxt(f'{path}.dat', delimiter='\t', usecols=[2, 3, 4, 7, 8, 9])

        # Assign channel data as NumPy arrays
        self.channel_1_time = data[:, 0]
        self.channel_1_energy = data[:, 1]
        self.channel_1_id = data[:, 2]

        self.channel_2_time = data[:, 3]
        self.channel_2_energy = data[:, 4]
        self.channel_2_id = data[:, 5]

    def get_channel_data(self, channel):
        """
        Returns the data for the specified channel (1 or 2).
        """
        if channel == 1:
            return self.channel_1_time, self.channel_1_energy, self.channel_1_id
        elif channel == 2:
            return self.channel_2_time, self.channel_2_energy, self.channel_2_id
        else:
            raise ValueError("Invalid channel number. Choose 1 or 2.")

    def get_valid_channel_pairs(self, min_count):
        """
        Returns valid (channel_1_id, channel_2_id) pairs that appear at least `min_count` times
        using only NumPy operations.

        Parameters:
        - min_count: The minimum count of occurrences for a (channel_1_id, channel_2_id) pair to be considered valid.

        Returns:
        - valid_pairs_list: A NumPy array of valid channel pairs where each element is [channel_1_id, channel_2_id].
        """
        # Stack channel_1_id and channel_2_id together into a 2D array
        stacked_array = np.vstack([self.channel_1_id, self.channel_2_id]).T

        # Find unique pairs and their counts
        unique_pairs, counts = np.unique(stacked_array, axis=0, return_counts=True)

        # Filter out the pairs that appear less than `min_count` times
        valid_pairs = unique_pairs[counts >= min_count]

        return valid_pairs

    def export_to_tsv(self, filename='output.tsv'):
        """
        Export the filtered channel data (time, energy, id) for both channels into a tab-separated values (TSV) file.
        """
        # Stack all the channel data into a single array
        combined_data = np.vstack([
            self.channel_1_time, self.channel_1_energy, self.channel_1_id,
            self.channel_2_time, self.channel_2_energy, self.channel_2_id
        ]).T

        # Save the combined data to a TSV file using NumPy's savetxt
        np.savetxt(filename, combined_data, delimiter='\t', header='channel_1_time\tchannel_1_energy\tchannel_1_id\tchannel_2_time\tchannel_2_energy\tchannel_2_id', fmt='%f')
        print(f"Data exported successfully to {filename}")

    def refactor(self, channel_1_time, channel_1_energy, channel_1_id, 
                 channel_2_time, channel_2_energy, channel_2_id):
        """
        Reinitialize the PETData class with provided channel data.
        """
        self.channel_1_time = channel_1_time
        self.channel_1_energy = channel_1_energy
        self.channel_1_id = channel_1_id

        self.channel_2_time = channel_2_time
        self.channel_2_energy = channel_2_energy
        self.channel_2_id = channel_2_id
