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
# csv_file = 'requested_CSVs/french_arthro_observations.csv'
# csv_output = 'biggest_french_member_by_obs.csv'
csv_file = 'requested_CSVs/south_american_arthro/south_american_arthro_observations_count.csv'
csv_output = 'requested_CSVs/south_american_arthro/biggest_member_by_obs.csv'
unique_column = 'family'
value_column = 'count'

result_set = find_biggest_value(csv_file, unique_column, value_column)
# Convert the result set to a DataFrame
result_df = pd.DataFrame(result_set, columns=['name', 'taxon_id', 'family', 'count'])

# Save the DataFrame as a CSV file
result_df.to_csv(csv_output, index=False, float_format='%.0f')