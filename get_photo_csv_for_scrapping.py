import sqlite3
import pandas as pd

sql_cmd_part1 = """SELECT taxon_id, photo_id, extension, photos.observation_uuid
FROM observations
INNER JOIN photos on photos.observation_uuid = observations.observation_uuid where taxon_id="""

sql_cmd_part2 = """;"""

# Open database and read list of species:
connection = sqlite3.connect("/mnt/disk1/datasets/iNaturalist/inat.db")
species = pd.read_csv('/mnt/disk1/datasets/iNaturalist/requested_CSVs/all_arthropods_families.csv', delimiter=',')

current_specie = 'A'  #TODO

# Loop through families:
print("Computing families starting with " + current_letter)
for index, row in species.iterrows():
    new_letter = row['name'][0]
    if new_letter != current_letter:
        current_letter = new_letter
        print("Computing families starting with " + current_letter)
    
    # print(index, row['name'], row['taxon_id'])

    # create sql command (FIND MEMBER OF FAMILY WITH MOST OBSERVATIONS):
    sql_command = sql_cmd_part1 + str(row['taxon_id']) + sql_cmd_part2

    # execute the statement
    db_df = pd.read_sql_query(sql_command, connection)
    if index == 0:
        db_df.to_csv('biggest_members.csv', index=False, header=True, mode='w')
    else:
        db_df.to_csv('biggest_members.csv', index=False, mode='a', header=False)


# close the connection
connection.close()
