import csv
import numpy as np
from tqdm import tqdm

def intersect_csvs(csv1_path, csv2_path, csv1_column, csv2_column, output_path):
    # Read the first CSV file
    csv1_data = np.genfromtxt(csv1_path, delimiter=',', dtype=None, names=True, encoding=None)

    # Read the second CSV file
    # csv2_data = np.genfromtxt(csv2_path, delimiter=';', dtype=None, names=True, encoding=None)
    csv2_data = np.genfromtxt(csv2_path, delimiter=',', dtype=None, names=True, encoding=None)

    print(csv1_data.dtype.names)
    print(csv2_data.dtype.names)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(csv1_data.dtype.names)  # Write the header row

    for row1 in tqdm(csv1_data):
        # # only start at a given row
        # if row1["name"] <= '"Micrepeira tubulofaciens"':
        #     continue
        for row2 in csv2_data:
            # if row1[csv1_column][1:-1] in row2[csv2_column][1:-1]:
            if row1[csv1_column] == row2[csv2_column]:
                with open(output_path, 'a') as f:
                    writer = csv.writer(f, delimiter=',', quotechar="'")
                    writer.writerow(row1)
                break


# Example usage
# csv1_path = 'requested_CSVs/all_arthropods_species.csv'
# csv2_path = '/home/eremy/Téléchargements/INPNrechercherrechercher_2024126.csv'
# csv1_column = 'name'
# csv2_column = 'Nom_valide'
# output_path = 'intersected_csv.csv'

# csv1_path = 'requested_CSVs/all_arthropods_species.csv'
# csv2_path = 'requested_CSVs/south_american_arthro/all_south_american_species.csv'
# csv1_column = 'taxon_id'
# csv2_column = 'taxon_id'
# output_path = 'requested_CSVs/south_american_arthro/all_south_american_arthro_species_tmp.csv'

# csv1_path = 'requested_CSVs/south_american_arthro/all_south_american_arthro_species.csv'
# csv2_path = 'requested_CSVs/all_french_arthropods.csv'
# csv1_column = 'taxon_id'
# csv2_column = 'taxon_id'
# output_path = 'requested_CSVs/south_american_arthro/all_south_american_arthro_species_intersect_test.csv'

csv1_path = 'requested_CSVs/south_american_arthro/south_american_arthro(no_french)hierarchy.csv'
csv2_path = 'requested_CSVs/biggest_french_member_by_obs_hierarchy.csv'
csv1_column = 'order'
csv2_column = 'order'
output_path = 'requested_CSVs/south_american_arthro/all_south_american_arthro(no_french)(only_french_orders).csv'

intersect_csvs(csv1_path, csv2_path, csv1_column, csv2_column, output_path)
