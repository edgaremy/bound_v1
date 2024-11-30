import sqlite3
import pandas as pd

# Open database and read list of species:
connection = sqlite3.connect("/mnt/disk1/datasets/iNaturalist/inat.db")
# species = pd.read_csv('requested_CSVs/biggest_french_member_by_obs.csv', delimiter=',')

# # Loop through species:
# for index, row in species.iterrows():
#     print("Current Specie: " + row['name'])
    
#     # print(index, row['name'], row['taxon_id'])

#     # create sql command (FIND ALL PHOTOS OF THE SPECIE):
#     sql_command = sql_cmd_part1 + str(row['taxon_id']) + sql_cmd_part2

#     # execute the statement
#     db_df = pd.read_sql_query(sql_command, connection)
#     if index == 0:
#         db_df.to_csv('photos_to_scrap.csv', index=False, header=True, mode='w')
#     else:
#         db_df.to_csv('photos_to_scrap.csv', index=False, mode='a', header=False)


# # close the connection
# connection.close()

# Returns a list of names as [class, order, family, genus, species]
def get_hierarchy_from_taxon_id(taxon_id):
    sql_command = f"SELECT name, ancestry FROM taxa WHERE taxon_id={taxon_id};"
    db_df = pd.read_sql_query(sql_command, connection)
    ancestry = [int(i) for i in db_df.iloc[0]['ancestry'].split('/')]

    hierarchy = ['','','','',db_df.iloc[0]['name']]

    for taxon in ancestry:
        sql_command = f"SELECT name, rank FROM taxa WHERE taxon_id={taxon};"
        db_df = pd.read_sql_query(sql_command, connection)
        rank = db_df.iloc[0]['rank']
        
        if rank == 'class':
            hierarchy[0] = db_df.iloc[0]['name']
        elif rank == 'order':
            hierarchy[1] = db_df.iloc[0]['name']
        elif rank == 'family':
            hierarchy[2] = db_df.iloc[0]['name']
        elif rank == 'genus':
            hierarchy[3] = db_df.iloc[0]['name']
            break

    return hierarchy

# Returns a list of names as [class, order, family, genus, species]
def get_hierarchy_from_name(specie_name):
    sql_command = f"SELECT taxon_id, rank FROM taxa WHERE name='{specie_name}' LIMIT 1;"
    db_df = pd.read_sql_query(sql_command, connection)
    taxon_id = db_df.iloc[0]['taxon_id']
    rank = db_df.iloc[0]['rank']

    if rank == 'species':
        return get_hierarchy_from_taxon_id(taxon_id)
    else:
        return get_hierarchy_from_taxon_id(taxon_id)

# Test the functions:
# print(get_hierarchy_from_taxon_id(473200))
# print(get_hierarchy_from_name("Aaroniella badonneli"))