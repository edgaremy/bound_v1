import os
import shutil

def flatten_folders(input_folder, output_directory):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate over all the files and folders in the input folder
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            # Get the full path of the file
            file_path = os.path.join(root, file)

            # Copy the file to the output directory
            shutil.copy(file_path, output_directory)


input_folder = "/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/DG"
output_directory = "/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/DG_flattened"

flatten_folders(input_folder, output_directory)