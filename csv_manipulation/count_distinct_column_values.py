import csv
import sys
from collections import defaultdict

def count_distinct_values(csv_file):
    distinct_counts = defaultdict(set)

    with open(csv_file, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            for column, value in row.items():
                distinct_counts[column].add(value)

    for column, values in distinct_counts.items():
        print(f"Column '{column}' has {len(values)} distinct values.")

# Example usage
csv_file = 'requested_CSVs/south_american_arthro/all_south_american_arthro(no_french_FAMILY)(only_french_orders).csv '
count_distinct_values(csv_file)
