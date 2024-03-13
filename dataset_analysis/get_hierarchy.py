import sqlite3
import pandas as pd

# Open database and read list of species:
connection = sqlite3.connect("/mnt/disk1/datasets/iNaturalist/inat.db")
# species = pd.read_csv('requested_CSVs/biggest_french_member_by_obs.csv', delimiter=',')

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
    try:
        sql_command = f"SELECT taxon_id, rank FROM taxa WHERE name='{specie_name}' LIMIT 1;"
        db_df = pd.read_sql_query(sql_command, connection)
        taxon_id = db_df.iloc[0]['taxon_id']
        rank = db_df.iloc[0]['rank']

        if rank == 'species':
            return get_hierarchy_from_taxon_id(taxon_id)
        else:
            return get_hierarchy_from_genus_name(specie_name.split(' ')[0], specie_name)
    except:
        return get_hierarchy_from_genus_name(specie_name.split(' ')[0], specie_name)

# Search directly for the genus name, if the specie name is not in the dataset
def get_hierarchy_from_genus_name(genus_name, specie_name):
    try:
        # Finding taxon_id of genus
        sql_command = f"SELECT taxon_id, rank FROM taxa WHERE name='{genus_name}' LIMIT 1;"
        db_df = pd.read_sql_query(sql_command, connection)
        taxon_id = db_df.iloc[0]['taxon_id']
        rank = db_df.iloc[0]['rank']

        # Get ancestry of genus
        sql_command = f"SELECT name, ancestry FROM taxa WHERE taxon_id={taxon_id};"
        db_df = pd.read_sql_query(sql_command, connection)
        ancestry = [int(i) for i in db_df.iloc[0]['ancestry'].split('/')]

        hierarchy = ['', '', '', genus_name, specie_name]

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
                break

        return hierarchy
    except:
        return [None, None, None, genus_name, specie_name]


# Test the functions:
# print(get_hierarchy_from_taxon_id(473200))
# print(get_hierarchy_from_name("Aaroniella badonneli"))