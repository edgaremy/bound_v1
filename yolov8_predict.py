from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO
import os
from collections import Counter

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

model = YOLO('runs/detect/train43/weights/best.pt') # load best.pt or last.pt of local model
# model.predict("/home/eremy/Téléchargements/4116123439.jpg", show=False, save=True, save_crop=True)

# Use the model
# model.train(data="/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD_1class/Bees_Detection(1class).yaml", epochs=100)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set

#path = model.export(format="onnx")

# im1 = "/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT1/Pictures/47219_37341.jpg"
# im2 = "/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT1/Pictures/47919_6925565.jpg"

# video = "/mnt/disk1/datasets/Bourdons/Select/BGOPR0593(cut).mp4"
# results = model.predict(video, stream=True, show=False, save=False, show_labels=False, show_conf=False)
# #model.track(video, show=True)#, save=False, show_labels=False, show_conf=False)

# # for r in results:
# #     if i == 0:
# #         print(r.names)
# #         print(r.boxes)
# #     i += 1
# names = model.names

# spieces = []

# for r in results:
#     for c in r.boxes.cls:
#         spieces.append(names[int(c)])

# species_count = Counter(spieces)
# sorted_species = sorted(species_count.items(), key=lambda x: x[1], reverse=True)

# for species, count in sorted_species:
#     print(f"{species}: {count}")


# # results = results.numpy()
# # r= results[0]
# # print(r.names)



# folder_path = "/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/DG"
# folder_path = "/mnt/disk1/datasets/Entomoscope/"
folder_path = "/mnt/disk1/datasets/iNaturalist/Arthropods/generalization/2nd_french_arthro/same_genus/tmp"

for folder_name in os.listdir(folder_path):
    subfolder_path = os.path.join(folder_path, folder_name)

    file_paths = []
    for root, dirs, files in os.walk(subfolder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

        extensions = set()
        for file_path in file_paths:
            file_name, file_extension = os.path.splitext(file_path)
            extensions.add(file_extension)

        print("List of file extensions:")
        print(list(set(extensions)))


    # print(len(file_paths))
    # file_paths.remove("/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT1/Pictures/1228962_168470622.gif")
    # print(len(file_paths))
    # # print(file_paths)

    chunk_size = 100
    num_chunks = len(file_paths) // chunk_size

    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        model.predict(file_paths[start:end], show=False, save=True, save_txt=True)

    remaining_files = len(file_paths) % chunk_size
    if remaining_files > 0:
        start = num_chunks * chunk_size
        end = start + remaining_files
        model.predict(file_paths[start:end], show=False, save=True, save_txt=True)