from get_hierarchy import get_hierarchy_from_taxon_id
import csv

def write_hierarchy_in_csv(input_csv, column_of_interest, output_csv):
    with open(input_csv, mode='r') as infile, open(output_csv, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)
        
        # Write header
        writer.writerow(['class', 'order', 'family', 'genus', 'specie'])
        
        for row in reader:
            taxon_id = row[column_of_interest]
            hierarchy = get_hierarchy_from_taxon_id(taxon_id)
            writer.writerow(hierarchy)

# Example usage:
# input_csv = 'requested_CSVs/south_american_arthro/all_south_american_arthro_species(no_french).csv'
# column_of_interest = 'taxon_id'
# output_csv = 'requested_CSVs/south_american_arthro/south_american_arthro(no_french)hierarchy.csv'
# write_hierarchy_in_csv(input_csv, column_of_interest, output_csv)

# input_csv = 'requested_CSVs/biggest_french_member_by_obs.csv'
# column_of_interest = 'taxon_id'
# output_csv = 'requested_CSVs/biggest_french_member_by_obs_hierarchy.csv'
# write_hierarchy_in_csv(input_csv, column_of_interest, output_csv)

input_csv = 'requested_CSVs/all_french_arthropods.csv'
column_of_interest = 'taxon_id'
output_csv = 'requested_CSVs/all_french_arthropods_hierarchy.csv'
write_hierarchy_in_csv(input_csv, column_of_interest, output_csv)
