import csv
import numpy as np
from tqdm import tqdm

# Subtract the rows of the SECOND CSV from the FIRST CSV
def subtract_csvs(csv1_path, csv2_path, csv1_column, csv2_column, output_path):
    # Read the first CSV file
    csv1_data = np.genfromtxt(csv1_path, delimiter=',', dtype=None, names=True, encoding=None)

    # Read the second CSV file
    csv2_data = np.genfromtxt(csv2_path, delimiter=',', dtype=None, names=True, encoding=None)

    print(csv1_data.dtype.names)
    print(csv2_data.dtype.names)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(csv1_data.dtype.names)  # Write the header row

    for row1 in tqdm(csv1_data):
        should_write = True
        for row2 in csv2_data:
            if row1[csv1_column] == row2[csv2_column]: # If the row is in the second CSV, do not copy
                should_write = False
                break
        if should_write:
            with open(output_path, 'a', newline='') as f: # copy every remaining rows
                writer = csv.writer(f, delimiter=',', quotechar="'")
                writer.writerow(row1)
                


# Example usage
csv1_path = 'requested_CSVs/south_american_arthro/all_south_american_arthro_species.csv'
csv2_path = 'requested_CSVs/all_french_arthropods.csv'
csv1_column = 'taxon_id'
csv2_column = 'taxon_id'
output_path = 'requested_CSVs/south_american_arthro/all_south_american_arthro_species(no_french).csv'

subtract_csvs(csv1_path, csv2_path, csv1_column, csv2_column, output_path)
