import pandas as pd

# Read the CSV file
data = pd.read_csv('/home/eremy/Téléchargements/observations-393029.csv')

# Group the data by taxon family and count the occurrences of each taxon species
grouped_data = data.groupby(['taxon_family_name', 'taxon_species_name']).size().reset_index(name='count')

# Find the most abundant taxon species for each taxon family
most_abundant_species = grouped_data.groupby('taxon_family_name').apply(lambda x: x.loc[x['count'].idxmax()])

# print(most_abundant_species)
# Print the results
for index, row in most_abundant_species.iterrows():
    print(f"Taxon Family: {row['taxon_family_name']}, Most Abundant Species: {row['taxon_species_name']}")


