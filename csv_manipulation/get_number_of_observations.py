# import csv
import sqlite3
import pandas as pd

# def process_csv(input_file, output_file):
#     # Create a connection to an in-memory SQLite database
#     conn = sqlite3.connect(':memory:')
#     cursor = conn.cursor()

#     # Create a table to store the CSV data
#     cursor.execute('CREATE TABLE data (line TEXT, result TEXT)')

#     # Read the input CSV file and process each line
#     with open(input_file, 'r') as csv_file:
#         reader = csv.reader(csv_file)
#         next(reader)  # Skip the header row

#         for row in reader:
#             line = ','.join(row)
#             sql = f"INSERT INTO data (line) VALUES ('{line}')"
#             cursor.execute(sql)

#     # Execute the SQL requests and store the results in the database
#     cursor.execute("UPDATE data SET result = 'Some result'")  # Replace with your SQL request

#     # Write the results to the output CSV file
#     with open(output_file, 'w', newline='') as csv_file:
#         writer = csv.writer(csv_file)
#         writer.writerow(['line', 'result'])  # Write the header row

#         cursor.execute('SELECT * FROM data')
#         for row in cursor.fetchall():
#             writer.writerow(row)

#     # Close the database connection
#     conn.close()

# # Usage example
# input_file = '/path/to/input.csv'
# output_file = '/path/to/output.csv'
# process_csv(input_file, output_file)

# Open database and read list of families:
connection = sqlite3.connect("/mnt/disk1/datasets/iNaturalist/inat.db")
species = pd.read_csv('requested_CSVs/all_french_arthropods.csv', delimiter=',')

current_letter = 'A'

# Loop through families:
print("Computing families starting with " + current_letter)
for index, row in species.iterrows():
    new_letter = row['name'][0]
    if new_letter != current_letter:
        current_letter = new_letter
        print("Computing species starting with " + current_letter)
    
    # print(index, row['name'], row['taxon_id'])

    # create sql command (FIND MEMBER OF FAMILY WITH MOST OBSERVATIONS):
    sql_command = f"SELECT name, t1.taxon_id, '{row['family']}' as family, COUNT(*) as count FROM taxa t1 JOIN observations o1 ON t1.taxon_id = o1.taxon_id WHERE t1.taxon_id={row['taxon_id']};"

    # execute the statement
    db_df = pd.read_sql_query(sql_command, connection)
    if index == 0:
        db_df.to_csv('french_arthro_observations.csv', index=False, header=True, mode='w')
    else:
        db_df.to_csv('french_arthro_observations.csv', index=False, mode='a', header=False)


# close the connection
connection.close()
