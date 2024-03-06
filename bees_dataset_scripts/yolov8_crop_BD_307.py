import csv
import os
import shutil
from ultralytics import YOLO

model = YOLO('runs/detect/train16/weights/best.pt') # load best.pt or last.pt of local model

def find_next_predict_number():
    found = False
    model_number = 2
    while not found:
        if os.path.exists('runs/detect/predict' + str(model_number)):
            model_number += 1
        else:
            found = True
    return model_number

def yolo_predict_and_crop(subfolder_path):

    file_paths = []
    for root, dirs, files in os.walk(subfolder_path):
        for file in files:
            file_extension = os.path.splitext(file)[1]
            if file_extension in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
            
    chunk_size = 100
    num_chunks = len(file_paths) // chunk_size
    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        try:
            model.predict(file_paths[start:end], save_crop=True, show=False, save=False, save_txt=False)
        except:
            print('A prediction error happened')
            continue

    remaining_files = len(file_paths) % chunk_size
    if remaining_files > 0:
        start = num_chunks * chunk_size
        end = start + remaining_files
        try:
            model.predict(file_paths[start:end], save_crop=True, show=False, save=False, save_txt=False)
        except:
            print('A prediction error happened')
            return
        model.predict(file_paths[start:end], save_crop=True, show=False, save=False, save_txt=False)


def crop_BD_307(csv_file, input_folder, output_folder):
    # Load classnames from CSV
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader) # Skip header
        classnames = list(reader)
        classnames = [x for xs in classnames for x in xs] # flatten list of lists

    next_predict_number = find_next_predict_number()
    print("The predict folder assumed is: predict" + str(next_predict_number))

    # Iterate over subfolders in input folders
    for input_folder in input_folders:
        for folder_name in os.listdir(input_folder):
            subfolder_path = os.path.join(input_folder, folder_name)

            # Check if subfolder is named after a classname
            if folder_name in classnames:
                # Apply function to subfolder
                yolo_predict_and_crop(subfolder_path)

                # Create output subfolder for classname
                output_subfolder = os.path.join(output_folder, folder_name)
                os.makedirs(output_subfolder, exist_ok=True)

                # Move results to output subfolder
                predict_path = 'runs/detect/predict' + str(next_predict_number) + '/crops/Bee'
                for file_name in os.listdir(predict_path):
                    file_path = os.path.join(predict_path, file_name)
                    shutil.move(file_path, output_subfolder)
    
    # remove empty predict directory
    shutil.rmtree('runs/detect/predict' + str(next_predict_number))

# Usage example
csv_file = 'bees_dataset_scripts/307_bee_species.csv'
input_folders = ['/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/inaturalist_24_01/Pictures',
                 '/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/whole_dataset/Anthophila',
                 '/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/whole_dataset/HS',
                 '/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/whole_dataset/Justine',
                 '/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/whole_dataset/LMDI',
                 '/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/whole_dataset/Spipoll']
output_folder = '/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/BD_307_cropped/ALL_else'
# input_folders = ['/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/whole_dataset/DG']
# output_folder = '/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/BD_307_cropped/DG'

crop_BD_307(csv_file, input_folders, output_folder)