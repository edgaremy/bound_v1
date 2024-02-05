import csv

def keep_each_element_number_n(input_file, output_file, column_name, n):
    lines_per_taxon = {}
    
    with open(input_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            taxon_id = row[column_name]
            if taxon_id not in lines_per_taxon:
                lines_per_taxon[taxon_id] = 1
            if lines_per_taxon[taxon_id] < n:
                    lines_per_taxon[taxon_id] += 1
            elif lines_per_taxon[taxon_id] == n:
                with open(output_file, 'a') as output:
                    writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
                    if output.tell() == 0:
                        writer.writeheader()
                    writer.writerow(row)
                    lines_per_taxon[taxon_id] += 1

# Usage example
input_file = 'requested_CSVs/photos_to_scrap.csv'
output_file = 'requested_CSVs/photos_to_scrap_NUMBER2.csv'
column_name = 'taxon_id'
number_to_keep = 2

keep_each_element_number_n(input_file, output_file, column_name, number_to_keep)
