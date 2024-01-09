import os
import csv

folder_path = '/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD_71_True_Annotations/vérifiés'  # Replace with the actual folder path

# Get the list of files in the folder
file_list = os.listdir(folder_path)

# Create a CSV file
csv_file_path = '/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD_71_True_Annotations/liste_classes_71.csv'  # Replace with the desired output CSV file path

with open(csv_file_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['species'])  # Write the header row

    # Write the file names to the CSV file
    file_list.sort()
    for file_name in file_list:

        writer.writerow([file_name.split('.')[0]])
