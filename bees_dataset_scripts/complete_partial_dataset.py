import os

def create_missing_folders(source_dir, target_dir):
    for folder in os.listdir(source_dir):
        source_folder_path = os.path.join(source_dir, folder)
        target_folder_path = os.path.join(target_dir, folder)
        
        if os.path.isdir(source_folder_path) and not os.path.exists(target_folder_path):
            os.makedirs(target_folder_path)
            print(f"Created folder: {target_folder_path}")

# Example usage
# source_directory = "/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/BD_307_cropped/dataset/train"
# target_directory = "/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/BD_307_cropped/dataset/test"
# create_missing_folders(source_directory, target_directory)

# Reverse effect
def delete_empty_folders(target_dir):
    for folder in os.listdir(target_dir):
        folder_path = os.path.join(target_dir, folder)
        if os.path.isdir(folder_path) and not os.listdir(folder_path):
            os.rmdir(folder_path)
            print(f"Deleted folder: {folder_path}")

# Example usage
# target_directory = "/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/BD_307_cropped/dataset/test"
# delete_empty_folders(target_directory)