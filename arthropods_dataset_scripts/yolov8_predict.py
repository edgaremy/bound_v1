from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO
import os
from collections import Counter

def find_last_predict_number():
    found = False
    model_number = 2
    while not found:
        if os.path.exists('runs/detect/predict' + str(model_number)):
            model_number += 1
        else:
            found = True
    return model_number - 1

def predict_images_with_yolov8(model_path, image_folder, output_folder=None):
    # Load a model
    # model = YOLO("yolov8n.yaml")  # build a new model from scratch
    # model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    model = YOLO(model_path) # load best.pt or last.pt of local model
    folder_path = image_folder

    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    #         extensions = set()
    # # for file_path in file_paths:
    # #     file_name, file_extension = os.path.splitext(file_path)
    # #     extensions.add(file_extension)

    # # print("List of file extensions:")
    # # print(list(set(extensions)))


    # print(len(file_paths))
    # file_paths.remove("/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT1/Pictures/1228962_168470622.gif")
    # print(len(file_paths))
    # # print(file_paths)

    chunk_size = 100
    num_chunks = len(file_paths) // chunk_size

    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        model.predict(file_paths[start:end], show=False, save=True, save_txt=True, verbose=False)

    remaining_files = len(file_paths) % chunk_size
    if remaining_files > 0:
        start = num_chunks * chunk_size
        end = start + remaining_files
        model.predict(file_paths[start:end], show=False, save=True, save_txt=True, verbose=False)

# Example Usage:
# model_path = "runs/detect/train25/weights/best.pt" # load best.pt or last.pt of local model
# folder_path = "/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT6/Pictures"
# predict_images_with_yolov8(model_path, folder_path)
        
# Example Usage:
# model_path = "runs/detect/train16/weights/best.pt" # load best.pt or last.pt of local model
# folder_path = ""
# predict_images_with_yolov8(model_path, folder_path)

# Example Usage:
model_path = "runs/detect/train40/weights/best.pt" # load best.pt or last.pt of local model
folder_path = "/mnt/disk1/datasets/iNaturalist/Arthropods/generalization/2nd_biggest_french_arthro/LIMIT1/Pictures"
predict_images_with_yolov8(model_path, folder_path)