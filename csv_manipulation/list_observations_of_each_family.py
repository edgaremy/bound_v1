import sqlite3
import pandas as pd


limit_obs = 50

# Open database and read list of families:
connection = sqlite3.connect("/mnt/disk1/datasets/iNaturalist/inat.db")
species = pd.read_csv('requested_CSVs/all_french_arthropod_families.csv', delimiter=',')

current_letter = 'A'

# Loop through families:
print("Computing families starting with " + current_letter)
for index, row in species.iterrows():
    new_letter = row['name'][0]
    if new_letter != current_letter:
        current_letter = new_letter
        print("Computing species starting with " + current_letter)
    
    # print(index, row['name'], row['taxon_id'])

    # create sql command (LIST 50 OBSERVATIONS OF THE FAMILY):
    sql_command = f"SELECT t1.taxon_id, '{row['taxon_id']}' as family, observation_uuid FROM observations o1 JOIN taxa t1 ON t1.taxon_id = o1.taxon_id WHERE rank = 'species' AND '/' || ancestry || '/' LIKE '%/{row['taxon_id']}/%' ORDER BY t1.taxon_id LIMIT {limit_obs};"

    # execute the statement
    db_df = pd.read_sql_query(sql_command, connection)
    if index == 0:
        db_df.to_csv('french_arthro_observations_list.csv', index=False, header=True, mode='w')
    else:
        db_df.to_csv('french_arthro_observations_list.csv', index=False, header=False, mode='a')


# close the connection
connection.close()
