import pandas as pd
from tqdm import tqdm

def get_next_biggest_specie_of_same_genus(first_specie_list, observation_counts_csv, taxonomy_csv, output_csv):
    # Load the CSV files into DataFrames
    species_interest_df = pd.read_csv(first_specie_list)
    observation_counts_df = pd.read_csv(observation_counts_csv)
    taxonomy_df = pd.read_csv(taxonomy_csv)
    
    # Create a list to store the results
    results = []
    
    # Iterate over each species of interest
    for index, row in tqdm(species_interest_df.iterrows()):
        specie_of_interest = row['specie']
        
        # Get the genus of the species of interest
        genus_of_interest = taxonomy_df[taxonomy_df['specie'] == specie_of_interest]['genus'].values[0]
        
        # Filter the observation_taxonomy_df to get species of the same genus
        same_genus_df = taxonomy_df[taxonomy_df['genus'] == genus_of_interest]
        
        merged_df = pd.merge(observation_counts_df, same_genus_df, left_on='name', right_on='specie')
        merged_df.rename(columns={'family_y': 'family'}, inplace=True)

        # Sort the merged dataframe by observation count in descending order
        sorted_df = merged_df.sort_values(by='count', ascending=False)
        
        # Find the next biggest specie of the same genus
        next_biggest_specie = None
        for i, r in sorted_df.iterrows():
            if r['specie'] != specie_of_interest:
                next_biggest_specie = r['specie']
                next_biggest_row = r
                break
        
        # Store the result in the list
        if next_biggest_specie:
            results.append(next_biggest_row[['class', 'order', 'family', 'genus', 'specie', 'taxon_id', 'count']])
    
    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(results)
    results_df['taxon_id'] = results_df['taxon_id'].astype(int) # make sure taxon_id is written as an integer
    
    # Save the results to a CSV file
    results_df.to_csv(output_csv, index=False)

# Example usage
first_specie_list = 'requested_CSVs/biggest_french_member_by_obs_hierarchy.csv'
observation_counts_csv = 'requested_CSVs/french_arthro_observations_count.csv'
taxonomy_csv = 'requested_CSVs/all_french_arthropods_hierarchy.csv'
output_csv = 'requested_CSVs/2nd_french_arthro/next_biggest_french_of_same_genus.csv'
get_next_biggest_specie_of_same_genus(first_specie_list, observation_counts_csv, taxonomy_csv, output_csv)