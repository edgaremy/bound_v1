import csv

def list_all_french_families_on_inat(file1, file2, output_file):
    # Read the first CSV file and convert it to a set
    with open(file1, 'r') as f1:
        csv_reader = csv.reader(f1)
        set1 = set()
        for row in csv_reader:
            set1.add(row[3])  # Assuming 'family' is the fourth column

    # Read the second CSV file and write matching lines to the third CSV file
    with open(file2, 'r') as f2, open(output_file, 'w', newline='') as f3:
        csv_reader = csv.reader(f2)
        csv_writer = csv.writer(f3)

        # Copy the headers from file2 to file3
        headers = next(csv_reader)
        csv_writer.writerow(headers)

        for row in csv_reader:
            if row[1] in set1:  # Assuming the family 'taxon_id' is the second column
                csv_writer.writerow(row)

# Usage example
list_all_french_families_on_inat('requested_CSVs/all_french_arthropods.csv',
                                 'requested_CSVs/all_arthropods_families.csv',
                                 'requested_CSVs/all_french_arthropod_families.csv')