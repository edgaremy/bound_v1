import csv

def keep_each_element_number_n_with_different_observation(input_file, output_file, n, column_name='taxon_id', column_name_2='observation_uuid'):
    lines_per_taxon = {}
    last_observation_per_taxon = {}
    
    with open(input_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            taxon_id = row[column_name]
            observation = row[column_name_2]
            
            if taxon_id not in lines_per_taxon:
                lines_per_taxon[taxon_id] = 1
                last_observation_per_taxon[taxon_id] = observation
            elif lines_per_taxon[taxon_id] < n and observation != last_observation_per_taxon[taxon_id]:
                lines_per_taxon[taxon_id] += 1
                last_observation_per_taxon[taxon_id] = observation
            elif lines_per_taxon[taxon_id] == n and observation != last_observation_per_taxon[taxon_id]:
                with open(output_file, 'a') as output:
                    writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
                    if output.tell() == 0:
                        writer.writeheader()
                    writer.writerow(row)
                    lines_per_taxon[taxon_id] += 1
                    # last_observation_per_taxon[taxon_id] = observation

# Usage example
# number_to_keep = 6
# input_file = 'requested_CSVs/photos_to_scrap.csv'
# output_file = 'requested_CSVs/photos_to_scrap_NUMBER'+ str(number_to_keep) +'.csv'
# column_name = 'taxon_id'
# column_name_2 = 'observation_uuid'

# keep_each_element_number_n_with_different_observation(input_file, output_file, number_to_keep, column_name, column_name_2)