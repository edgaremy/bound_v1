import csv

def limit_number_of_csv_lines(input_file, output_file, column_name, max_lines):
    lines_per_taxon = {}
    
    with open(input_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            taxon_id = row[column_name]
            if taxon_id not in lines_per_taxon:
                lines_per_taxon[taxon_id] = 0
            if lines_per_taxon[taxon_id] < max_lines:
                with open(output_file, 'a') as output:
                    writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
                    if output.tell() == 0:
                        writer.writeheader()
                    writer.writerow(row)
                    lines_per_taxon[taxon_id] += 1

# Usage example
input_file = 'requested_CSVs/photos_to_scrap.csv'
output_file = 'requested_CSVs/photos_to_scrap_LIMIT50.csv'
column_name = 'taxon_id'
max_lines = 50

limit_number_of_csv_lines(input_file, output_file, column_name, max_lines)
