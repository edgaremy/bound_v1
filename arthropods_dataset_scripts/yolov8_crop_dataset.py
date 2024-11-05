import os
import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("input_folder", type=str)
parser.add_argument("-o", "--output", type=str, default="")
parser.add_argument("-b", "--batch_size", type=int, default=100)

input_folder = parser.parse_args().input_folder
output_folder = parser.parse_args().output
if output_folder == "":
    output_folder = os.path.join(input_folder, 'cropped')
batch_size = parser.parse_args().batch_size

model = YOLO('runs/detect/train40/weights/best.pt') # load best.pt or last.pt of local model

def yolo_predict_and_crop(subfolder_path, output, chunk_size=100):

    file_paths = []
    for root, dirs, files in os.walk(subfolder_path):
        for file in files:
            file_extension = os.path.splitext(file)[1]
            if file_extension in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
            
    num_chunks = len(file_paths) // chunk_size
    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        try:
            model.predict(file_paths[start:end], save_crop=True, show=False, save=False, save_txt=False, project=output)
        except:
            print('A prediction error happened')
            continue

    remaining_files = len(file_paths) % chunk_size
    if remaining_files > 0:
        start = num_chunks * chunk_size
        end = start + remaining_files
        try:
            model.predict(file_paths[start:end], save_crop=True, show=False, save=False, save_txt=False, project=output)
        except:
            print('A prediction error happened')
            return
        model.predict(file_paths[start:end], save_crop=True, show=False, save=False, save_txt=False, project=output)

yolo_predict_and_crop(input_folder, output_folder, batch_size)