import sqlite3
import pandas as pd

# SQL command parts
sql_cmd_begin = """
SELECT name, t1.taxon_id, ancestry, COUNT(*) as count
FROM taxa t1
JOIN observations o1 ON t1.taxon_id = o1.taxon_id
WHERE rank = 'species'
AND '/' || ancestry || '/' LIKE '%/"""

sql_cmd_end = """/%'
GROUP BY name, t1.taxon_id
ORDER BY count DESC
LIMIT 1;
"""

# Open database and read list of families:
connection = sqlite3.connect("/mnt/disk1/datasets/iNaturalist/inat.db")
families = pd.read_csv('/mnt/disk1/datasets/iNaturalist/requested_CSVs/all_arthropods_families.csv', delimiter=',')

current_letter = 'A'

# Loop through families:
print("Computing families starting with " + current_letter)
for index, row in families.iterrows():
    new_letter = row['name'][0]
    if new_letter != current_letter:
        current_letter = new_letter
        print("Computing families starting with " + current_letter)
    
    # print(index, row['name'], row['taxon_id'])

    # create sql command (FIND MEMBER OF FAMILY WITH MOST OBSERVATIONS):
    sql_command = sql_cmd_begin + str(row['taxon_id']) + sql_cmd_end

    # execute the statement
    db_df = pd.read_sql_query(sql_command, connection)
    if index == 0:
        db_df.to_csv('biggest_members.csv', index=False, header=True, mode='w')
    else:
        db_df.to_csv('biggest_members.csv', index=False, mode='a', header=False)


# close the connection
connection.close()
