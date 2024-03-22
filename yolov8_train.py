from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
model = YOLO('runs/detect/train16/weights/best.pt') # load best.pt or last.pt of local model


# Use the model
# model.train(data="/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD_1class/Bees_Detection(1class).yaml", epochs=100)  # train the model

# Evaluate model performance on the validation set
# metrics = model.val()
metrics = model.val(data="/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/DG_evaluate_predictions/dataset/DG_Detection.yaml")

#path = model.export(format="onnx")