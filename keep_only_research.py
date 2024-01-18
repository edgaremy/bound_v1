import csv

input_file = "/mnt/disk1/datasets/iNaturalist/all-inat_CSVs/observations.csv"
output_file = "/mnt/disk1/datasets/iNaturalist/all-inat_CSVs/observations_research_only.csv"

with open(input_file, "r") as file_in, open(output_file, "w", newline="") as file_out:
    reader = csv.DictReader(file_in, delimiter="\t")
    writer = csv.DictWriter(file_out, delimiter="\t", fieldnames=reader.fieldnames)
    writer.writeheader()

    for row in reader:
        if row["quality_grade"] == "research":
            writer.writerow(row)
