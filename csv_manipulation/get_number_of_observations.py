import sqlite3
import pandas as pd

# input_csv = 'requested_CSVs/all_french_arthropods.csv'
# output_csv = 'french_arthro_observations_count.csv'
# column_of_interest = 'name'

input_csv = 'requested_CSVs/south_american_arthro/all_south_american_arthro(no_french_FAMILY)(only_french_orders).csv'
output_csv = 'requested_CSVs/south_american_arthro/south_american_arthro(only_french_orders)observations_count.csv'
column_of_interest = 'specie'

# Open database and read list of families:
connection = sqlite3.connect("/mnt/disk1/datasets/iNaturalist/inat.db")
species = pd.read_csv(input_csv, delimiter=',')

current_letter = 'A'

# Loop through families:
print("Computing families starting with " + current_letter)
for index, row in species.iterrows():
    new_letter = row[column_of_interest][0]
    if new_letter != current_letter:
        current_letter = new_letter
        print("Computing species starting with " + current_letter)
    
    # print(index, row['name'], row['taxon_id'])

    # create sql command (FIND MEMBER OF FAMILY WITH MOST OBSERVATIONS):
    sql_command = f"SELECT name, t1.taxon_id, '{row['family']}' as family, COUNT(*) as count FROM taxa t1 JOIN observations o1 ON t1.taxon_id = o1.taxon_id WHERE t1.taxon_id={row['taxon_id']};"

    # execute the statement
    db_df = pd.read_sql_query(sql_command, connection)
    if index == 0:
        db_df.to_csv(output_csv, index=False, header=True, mode='w')
    else:
        db_df.to_csv(output_csv, index=False, header=False, mode='a')


# close the connection
connection.close()
