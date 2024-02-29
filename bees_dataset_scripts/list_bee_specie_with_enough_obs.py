import csv

def copy_values(input_file, output_file, check_column, threshold, copy_column):
    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        next(reader) # skip header
        rows = list(reader)


    output_rows = []
    for row in rows:
        if float(row[check_column]) >= threshold:
            #row[copy_column] = row[check_column]
            output_rows.append([row[copy_column]])

    # Sort each line by the first column (species name)
    output_rows.sort(key=lambda x: x[0])
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(output_rows)

# Example usage
input_file = 'bees_dataset_scripts/0713_nb_photos_espèces_bdd.csv'
output_file = '/home/eremy/Documents/CODE/bound_v1/bees_dataset_scripts/species_names.csv'
check_column = 9  # Column index to check for threshold
threshold = 30  # Threshold value
copy_column = 1  # Column index to copy value from

copy_values(input_file, output_file, check_column, threshold, copy_column)