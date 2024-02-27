from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
model = YOLO('runs/detect/train16/weights/best.pt') # load best.pt or last.pt of local model


# Use the model
model.train(data="/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT5/dataset/Arthropods_LIMIT5.yaml", epochs=100)  # train the model

# Evaluate model performance on the validation set
metrics = model.val()
# metrics = model.val(data="/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT2/dataset/Arthropods_LIMIT2.yaml")

#path = model.export(format="onnx")