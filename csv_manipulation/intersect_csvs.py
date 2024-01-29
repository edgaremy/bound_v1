import csv
import numpy as np
from tqdm import tqdm

def intersect_csvs(csv1_path, csv2_path, csv1_column, csv2_column, output_path):
    # Read the first CSV file
    csv1_data = np.genfromtxt(csv1_path, delimiter=',', dtype=None, names=True, encoding=None)

    # Read the second CSV file
    csv2_data = np.genfromtxt(csv2_path, delimiter=';', dtype=None, names=True, encoding=None)

    print(csv1_data.dtype.names)
    print(csv2_data.dtype.names)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(csv1_data.dtype.names)  # Write the header row

    # Find the common values in the given columns
    # i = 0
    common_values = np.array([], dtype=csv1_data[csv1_column].dtype)
    for row1 in tqdm(csv1_data):
        # i += 1
        # if i == 50:
        #     break
        for row2 in csv2_data:
            if row1[csv1_column][1:-1] in row2[csv2_column][1:-1]:
                # print(row1[csv1_column])
                with open(output_path, 'a') as f:
                    writer = csv.writer(f, delimiter=',', quotechar="'")
                    writer.writerow(row1)
                break


# Example usage
csv1_path = 'requested_CSVs/all_arthropods_species.csv'
csv2_path = '/home/eremy/Téléchargements/INPNrechercherrechercher_2024126.csv'
csv1_column = 'name'
csv2_column = 'Nom_valide'
output_path = 'intersected_csv.csv'

intersect_csvs(csv1_path, csv2_path, csv1_column, csv2_column, output_path)
