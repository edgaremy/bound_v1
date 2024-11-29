import csv
import sqlite3
import pandas as pd

# Chooses the appropriate photo for each taxon_id
# Each time the photo will come from a different observation
# If there is no more observation, the function will find a photo from another given list
def keep_each_element_number_n_with_different_observation(input_file, input_file_2, family_file, output_file, n):
    lines_per_taxon = {}
    last_observation_per_taxon = {}
    
    # Open database:
    connection = sqlite3.connect("/mnt/disk1/datasets/iNaturalist/inat.db")

    with open(input_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            taxon_id = row['taxon_id']
            observation = row['observation_uuid']
            
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

        # For each encountered taxon_id, if no photo was found, we will take a photo from the second list
        for taxon_id in lines_per_taxon:
            if lines_per_taxon[taxon_id] < n+1: # There wasn't enough photos (from different observations) for this taxon_id
                family = None
                # Find the family of the taxon_id
                with open(family_file, 'r') as file_3:
                    reader_3 = csv.DictReader(file_3)
                    for row in reader_3:
                        if row['taxon_id'] == taxon_id:
                            family = row['family']
                            break

                # Take the next available photo from the second list
                with open(input_file_2, 'r') as file_2:
                    reader_2 = csv.DictReader(file_2)
                    for row in reader_2:
                        if row['family'] == family and row['taxon_id'] != taxon_id:
                            if lines_per_taxon[taxon_id] == n:
                                # Get first photo from the observation:
                                sql_command = f"SELECT taxon_id, photo_id, extension, p1.observation_uuid FROM observations o1 INNER JOIN photos p1 on p1.observation_uuid = o1.observation_uuid WHERE o1.observation_uuid='{row['observation_uuid']}' LIMIT 1;"
                                db_df = pd.read_sql_query(sql_command, connection)

                                # Write the photo to the output file:
                                with open(output_file, 'a') as output:
                                    writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
                                    if output.tell() == 0:
                                        writer.writeheader()
                                    writer.writerow({'taxon_id': db_df.iloc[0]['taxon_id'], 'photo_id': db_df.iloc[0]['photo_id'], 'extension': db_df.iloc[0]['extension'], 'observation_uuid': db_df.iloc[0]['observation_uuid']})
                                break
                            else:
                                lines_per_taxon[taxon_id] += 1



# Usage example
# number_to_keep = 1
# input_file = 'requested_CSVs/photos_to_scrap.csv'
# input_file_2 = 'requested_CSVs/french_arthro_observations_list.csv'
# family_file = 'requested_CSVs/biggest_french_member_by_obs.csv'
# output_file = 'requested_CSVs/photos_to_scrap_NUMBER'+ str(number_to_keep) +'.csv'

# number_to_keep = 1
# input_file = 'requested_CSVs/2nd_french_arthro/photos_to_scrap_2nd.csv'
# input_file_2 = 'requested_CSVs/french_arthro_observations_list.csv'
# family_file = 'requested_CSVs/2nd_french_arthro/2nd_biggest_french_member_by_obs.csv'
# output_file = 'requested_CSVs/2nd_french_arthro/photos_to_scrap_2nd_NUMBER'+ str(number_to_keep) +'.csv'
# keep_each_element_number_n_with_different_observation(input_file, input_file_2, family_file, output_file, number_to_keep)