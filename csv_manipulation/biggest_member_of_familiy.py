import sqlite3
import pandas as pd
import pandas as pd


def find_biggest_value(csv_file, unique_column, value_column):
    # Read the CSV file as a DataFrame
    df = pd.read_csv(csv_file)

    # Group the DataFrame by the unique column
    grouped = df.groupby(unique_column)

    # Initialize an empty set to store the rows with the biggest values
    result = set()

    # Iterate over each group
    for group_name, group_df in grouped:
        # Find the row with the biggest value in the value column
        max_value = group_df[value_column].max()
        if max_value > 0:
            max_row = group_df.loc[group_df[value_column].idxmax()]

            # Add the row to the result set
            result.add(tuple(max_row))

    result = sorted(result, key=lambda x: x[0], reverse=False)

    return result

# Example usage
csv_file = 'requested_CSVs/french_arthro_observations.csv'
unique_column = 'family'
value_column = 'count'

result_set = find_biggest_value(csv_file, unique_column, value_column)
# print(result_set)
# Convert the result set to a DataFrame
result_df = pd.DataFrame(result_set, columns=['name', 'taxon_id', 'family', 'count'])

# Save the DataFrame as a CSV file
result_df.to_csv('biggest_french_member_by_obs.csv', index=False, float_format='%.0f')



# sort_by = 'photos_count' # 'observations_count' or 'photos_count'

# # SQL command parts
# if sort_by == 'observations_count':
#     sql_cmd_part1 = """SELECT name, t1.taxon_id, '"""

#     sql_cmd_part2 = """' as family, COUNT(*) as count
# FROM taxa t1
# JOIN observations o1 ON t1.taxon_id = o1.taxon_id
# WHERE rank = 'species'
# AND '/' || ancestry || '/' LIKE '%/"""

#     sql_cmd_part3 = """/%'
# GROUP BY name, t1.taxon_id
# ORDER BY count DESC
# LIMIT 1;
#     """

# elif sort_by == 'photos_count':
#     sql_cmd_part1 = """SELECT name, t1.taxon_id, '"""

#     sql_cmd_part2 = """' as family, COUNT(*) as count
# FROM taxa t1
# JOIN observations o1 ON t1.taxon_id = o1.taxon_id
# JOIN photos p1 ON p1.observation_uuid = o1.observation_uuid
# WHERE rank = 'species'
# AND '/' || ancestry || '/' LIKE '%/"""

#     sql_cmd_part3 = """/%'
# GROUP BY name, t1.taxon_id
# ORDER BY count DESC
# LIMIT 1;
#     """

# # Open database and read list of families:
# connection = sqlite3.connect("/mnt/disk1/datasets/iNaturalist/inat.db")
# families = pd.read_csv('/mnt/disk1/datasets/iNaturalist/requested_CSVs/all_arthropods_families.csv', delimiter=',')

# current_letter = 'A'

# # Loop through families:
# print("Computing families starting with " + current_letter)
# for index, row in families.iterrows():
#     new_letter = row['name'][0]
#     if new_letter != current_letter:
#         current_letter = new_letter
#         print("Computing families starting with " + current_letter)
    
#     # print(index, row['name'], row['taxon_id'])

#     # create sql command (FIND MEMBER OF FAMILY WITH MOST OBSERVATIONS):
#     sql_command = sql_cmd_part1 + str(row['taxon_id']) + sql_cmd_part2 + str(row['taxon_id']) + sql_cmd_part3

#     # execute the statement
#     db_df = pd.read_sql_query(sql_command, connection)
#     if index == 0:
#         db_df.to_csv('biggest_members.csv', index=False, header=True, mode='w')
#     else:
#         db_df.to_csv('biggest_members.csv', index=False, mode='a', header=False)


# # close the connection
# connection.close()
