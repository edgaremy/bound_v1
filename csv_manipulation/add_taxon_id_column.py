import sqlite3
import pandas as pd

# Open database and read list of species:
connection = sqlite3.connect("/mnt/disk1/datasets/iNaturalist/inat.db")

def get_taxon_id_from_name(specie_name):
    sql_command = f"SELECT taxon_id, rank FROM taxa WHERE name='{specie_name}' LIMIT 1;"
    db_df = pd.read_sql_query(sql_command, connection)
    taxon_id = db_df.iloc[0]['taxon_id']
    return taxon_id

def add_taxon_id_column(csv_file_path, column_name):
    df = pd.read_csv(csv_file_path)
    df['taxon_id'] = df[column_name].apply(get_taxon_id_from_name)
    df.to_csv(csv_file_path, index=False)

# Example usage:
csv_file_path = 'requested_CSVs/south_american_arthro/all_south_american_arthro(no_french)(only_french_orders) copy.csv'
column_name = 'specie'  # name column, used to get the taxon_id
add_taxon_id_column(csv_file_path, column_name)