import csv

# Compare 2 csv files and print the differences
# Compares only the values at the column index specified
def compare_csv_files(file1, file2, column):
    with open(file1, 'r') as csv_file1, open(file2, 'r') as csv_file2:
        reader1 = csv.reader(csv_file1)
        reader2 = csv.reader(csv_file2)
        
        count = 0
        count_total = 0

        for row1, row2 in zip(reader1, reader2):
            count_total += 1
            if row1[column] != row2[column]:
                count += 1
                print(f"Difference found in column {column}:")
                print(f"File 1: {row1}")
                print(f"File 2: {row2}")
                print()
    print(f"Total differences found: {count}/{count_total}")

# Specify the file paths and column index to compare
file1 = 'requested_CSVs/biggest_members_by_observations.csv'
file2 = 'requested_CSVs/biggest_members_by_photos.csv'
column_to_compare = 0

compare_csv_files(file1, file2, column_to_compare)
