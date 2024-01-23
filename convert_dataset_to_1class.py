import os
import shutil

# Path to the label folder
label_folder = '/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD_71_visited/labels'

# Path to the destination folder
destination_folder = '/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD_1class/labels'

# Iterate through each folder inside the label folder
for folder_name in ['train', 'test', 'val']:
    folder_path = os.path.join(label_folder, folder_name)

    # Iterate through each label file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            # Read the contents of the label file
            with open(os.path.join(folder_path, filename), 'r') as file:
                lines = file.readlines()

            # Modify the class number in each line to 0
            modified_lines = [line.replace(line.split()[0], '0') for line in lines]

            # Write the modified contents to a new label file in the destination folder
            with open(os.path.join(destination_folder, folder_name, filename), 'w') as file:
                file.writelines(modified_lines)
