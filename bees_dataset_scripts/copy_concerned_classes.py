import csv
import os
import shutil

def copy_concerned_classes(csv_file, input_folders, output_folder):
    # Load classnames from CSV
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader) # Skip header
        classnames = list(reader)
        classnames = [x for xs in classnames for x in xs] # flatten list of lists
    

    # Iterate over subfolders in input folders
    for input_folder in input_folders:
        for folder_name in os.listdir(input_folder):
            subfolder_path = os.path.join(input_folder, folder_name)

            # Check if subfolder is named after a classname
            if folder_name in classnames:
                # Copy subfolder to output folder
                output_subfolder_path = os.path.join(output_folder, folder_name)
                shutil.copytree(subfolder_path, output_subfolder_path)
    
# Example Usage:
csv_file = 'bees_dataset_scripts/307_bee_species.csv'
input_folders = ['/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/whole_dataset/DG']
output_folder = '/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/DG'

copy_concerned_classes(csv_file, input_folders, output_folder)