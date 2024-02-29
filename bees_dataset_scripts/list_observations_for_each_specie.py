import csv
import sqlite3
import pandas as pd

input_file = "bees_dataset_scripts/307_bee_species.csv"
output_file = "/home/eremy/Documents/CODE/bound_v1/bees_dataset_scripts/every_species_observations_LIMIT500.csv"

# Open database:
connection = sqlite3.connect("/mnt/disk1/datasets/iNaturalist/inat.db")

with open(input_file, 'r') as species:
    with open(output_file, 'a') as output:
        reader = csv.DictReader(species)
        writer = csv.DictWriter(output, fieldnames=['name', 'taxon_id', 'observation_uuid'])
        if output.tell() == 0:
            writer.writeheader()

        for row in reader:
            sql_command = f"SELECT name, t1.taxon_id, observation_uuid FROM taxa t1 JOIN observations o1 ON t1.taxon_id = o1.taxon_id WHERE t1.name='{row['name']}' LIMIT 500;"
            db_df = pd.read_sql_query(sql_command, connection)
            # Write the photos to the output file if they exist:
            for index, row in db_df.iterrows():
                writer.writerow({'name': row['name'], 'taxon_id': row['taxon_id'], 'observation_uuid': row['observation_uuid']})
            