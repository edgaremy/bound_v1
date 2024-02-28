from ultralytics import YOLO
import os
import re

# Find the next available train model number in the run/detect folder
def find_last_model_number():
    found = False
    model_number = 2
    while not found:
        if os.path.exists('runs/detect/train' + str(model_number)):
            model_number += 1
        else:
            found = True
    return model_number - 1

def train_yolov8(yaml_path, epoch=100):
    # Load a model
    # model = YOLO("yolov8n.yaml")  # build a new model from scratch
    # model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    model = YOLO('runs/detect/train16/weights/best.pt') # load best.pt or last.pt of local model


    # Use the model
    model.train(data=yaml_path, epochs=epoch)  # train the model

    # Evaluate model performance on the validation set
    metrics = model.val()
    # metrics = model.val(data="/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT2/dataset/Arthropods_LIMIT2.yaml")

    #path = model.export(format="onnx")