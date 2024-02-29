import csv
import sqlite3
import pandas as pd

input_file = "/home/eremy/Documents/CODE/bound_v1/bees_dataset_scripts/every_species_observations_LIMIT300.csv"
output_file = "/home/eremy/Documents/CODE/bound_v1/bees_dataset_scripts/every_species_firstphoto_LIMIT300.csv"

# Open database:
connection = sqlite3.connect("/mnt/disk1/datasets/iNaturalist/inat.db")

with open(input_file, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
                # Get first photo from the observation:
                sql_command = f"SELECT taxon_id, photo_id, extension, p1.observation_uuid FROM observations o1 INNER JOIN photos p1 on p1.observation_uuid = o1.observation_uuid WHERE o1.observation_uuid='{row['observation_uuid']}' LIMIT 1;"
                db_df = pd.read_sql_query(sql_command, connection)

                # Write the photo to the output file:
                with open(output_file, 'a') as output:
                    writer = csv.DictWriter(output, fieldnames=['taxon_id', 'photo_id', 'extension', 'observation_uuid'])
                    if output.tell() == 0:
                        writer.writeheader()
                    writer.writerow({'taxon_id': db_df.iloc[0]['taxon_id'], 'photo_id': db_df.iloc[0]['photo_id'], 'extension': db_df.iloc[0]['extension'], 'observation_uuid': db_df.iloc[0]['observation_uuid']})