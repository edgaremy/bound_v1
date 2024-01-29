import csv
from tqdm import tqdm

def find_lines(csv1_path, csv2_path, csv1_column, csv2_column, csv_out_path):

    # store the lines of CSV1 in a list, as each line wil be read multiple times
    csv1_lines = []
    with open(csv1_path, 'r') as csv1_file:
        csv1_reader = csv.reader(csv1_file)
        next(csv1_reader, None)  # skip the headers
        for line in csv1_reader:
                csv1_lines.append(line)

    # Read CSV2 and get the IDs from the given column
    csv2_lines = []
    with open(csv2_path, 'r') as csv2_file:
        csv2_reader = csv.reader(csv2_file)
        next(csv2_reader, None)  # skip the headers
        for line in csv2_reader:
                csv2_lines.append(line)

    # # Find lines in CSV1 where the ID is part of the given column
    # matching_lines = []
    # for line in csv1_lines:
    #     if any(csv2_id in line[csv1_column] for csv2_id in csv2_ids):
    #         matching_lines.append(line)

    with open(csv_out_path, "w") as outfile:

            csv_writer = csv.DictWriter(outfile, fieldnames = ['name', 'taxon_id', 'ancestry', 'family'])
            csv_writer.writeheader()

            for specie_line in tqdm(csv1_lines):
                for family_line in csv2_lines:
                    if '/' + family_line[csv2_column] + '/' in specie_line[csv1_column]:
                        csv_writer.writerow({'name': specie_line[0], 'taxon_id': specie_line[1], 'ancestry': specie_line[2], 'family': family_line[csv2_column]})
                        break

    return

# Usage example
csv1_path = 'requested_CSVs/intersected_csv.csv'
csv2_path = 'requested_CSVs/all_arthropods_families.csv'
csv1_column = 2  # Index of the column ancestry in CSV1
csv2_column = 1  # Index of the column taxon_id in CSV2
csv_out_path = 'all_french_arthropods.csv'
matching_lines = find_lines(csv1_path, csv2_path, csv1_column, csv2_column, csv_out_path)
print(matching_lines)
