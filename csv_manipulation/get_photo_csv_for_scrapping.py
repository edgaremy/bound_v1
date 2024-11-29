import sqlite3
import pandas as pd

sql_cmd_part1 = """SELECT taxon_id, photo_id, extension, photos.observation_uuid
FROM observations
INNER JOIN photos on photos.observation_uuid = observations.observation_uuid where taxon_id="""

add_location_criteria = False
if add_location_criteria:
    sql_cmd_location = """ AND -55 < observations.latitude AND observations.latitude < 15
AND -85 < observations.longitude AND observations.longitude < -30"""
else:
    sql_cmd_location = """"""

limit_of_observations_to_keep = 1 # set to -1 to keep all observations
if limit_of_observations_to_keep == -1:
    sql_cmd_part2 = """;"""
else:
    sql_cmd_part2 = """ LIMIT """ + str(limit_of_observations_to_keep) + """;"""



# Open database and read list of species:
connection = sqlite3.connect("/mnt/disk1/datasets/iNaturalist/inat.db")

# species = pd.read_csv('requested_CSVs/biggest_french_member_by_obs.csv', delimiter=',')
# output_csv = 'requested_CSVs/photos_to_scrap.csv'
# species = pd.read_csv('requested_CSVs/2nd_french_arthro/next_biggest_french_genus.csv', delimiter=',')
# output_csv = 'requested_CSVs/2nd_french_arthro/photos_to_scrap_next_genus_LIMIT1.csv'
species = pd.read_csv('requested_CSVs/2nd_french_arthro/next_biggest_french_of_same_genus.csv', delimiter=',')
output_csv = 'requested_CSVs/2nd_french_arthro/photos_to_scrap_same_genus_LIMIT1.csv'
# species = pd.read_csv('requested_CSVs/south_american_arthro/biggest_member_by_obs.csv', delimiter=',')
# output_csv = 'requested_CSVs/south_american_arthro/photos_to_scrap_LIMIT1.csv'

start_with = None # set to None to start from the beginning, set to a taxon_id to start from that taxon_id

# Loop through species:
for index, row in species.iterrows():
    if start_with is not None:
        if row['taxon_id'] != start_with:
            continue
        else:
            start_with = None
    # print("Current Specie: " + row['name'])
    print("Current Specie: " + row['specie'])
    
    # print(index, row['name'], row['taxon_id'])

    # create sql command (FIND ALL PHOTOS OF THE SPECIE):
    sql_command = sql_cmd_part1 + str(row['taxon_id']) + sql_cmd_location+ sql_cmd_part2

    # execute the statement
    db_df = pd.read_sql_query(sql_command, connection)
    if index == 0:
        db_df.to_csv(output_csv, index=False, header=True, mode='w')
    else:
        db_df.to_csv(output_csv, index=False, mode='a', header=False)


# close the connection
connection.close()
