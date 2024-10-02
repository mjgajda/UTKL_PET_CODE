import concurrent
from ..PET_CLASSES.Energy import *
from ..PET_CLASSES.CTR import CTR
from ..PET_CLASSES.Coinc import Coinc
from ..PET_CLASSES.PET_Data import *

# Define file names
def builder(paths,minCoinc):

    def process_pair(pair, petData):
        """
        Process each valid channel pair, applying energy cuts and time cuts,
        and return the result if successful.
        """
        try:
            print(pair)

            # Create a coincidence pair object to manipulate
            channelPair = Coinc(petData, pair[0], pair[1])
            print(petData.channel_1_energy)
            petData = None

            # Make an energy spectra object from the coincidence object
            print('energy cut')
            energySpectra = Energy(channelPair)
            
            # Attempt to fit the energy spectra, specify the number of sigma for the fit and
            # the number of events needed to be recorded
            result = energySpectra.energyCuts(numberSigma=3.5, threshold=20)

            # If energy cut fails, skip this pair
            if result is None:
                return None

            # Make the CTR of the observed pair
            print('time cut')
            CTRDistribution = CTR(result)

            # Fit to the time distribution
            result = CTRDistribution.timeCut(numberSigma=3.5, threshold=20)

            # Return the result if successful
            if result is not None:
                return result
            else:
                return None

        except Exception as e:
            print(f"Error processing pair {pair}: {e}")
            return None


    # Main loop through all the names in the function
    if __name__ == '__main__':
        
        for path in paths:
            # Load the whole PET data object
            print('Getting Data')
            petData = PETData(path)

            # Filter out the channel pairs less than n and get the valid pairs
            print('Filtering by number of coincidences')
            validPairs = petData.get_valid_channel_pairs(minCoinc)

            # Process all valid pairs in parallel using ProcessPoolExecutor
            print('Processing pairs in parallel')
            with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:  # Adjust 'max_workers' to control parallelism
                # Submit tasks to the executor
                futures = [executor.submit(process_pair, pair, petData) for pair in validPairs]
                
                # Gather results as they are completed
                results = []
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result is not None:  # Skip failed tasks
                        results.append(result)

            # Filter out None results (failed fits)
            filtered = [result for result in results if result is not None]

            # If we have any filtered results, proceed with further processing
            if filtered:
                # Refactor the PET data object with the filtered data
                print('Refactoring PET data with filtered results')
                petData.refactor(coinc.sum_coinc_data(filtered))

                # Export the results to a TSV file
                print('Exporting data to TSV')
                petData.export_to_tsv()
