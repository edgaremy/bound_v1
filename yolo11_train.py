from ultralytics import YOLO

# Load a model
# model = YOLO("yolov11n.yaml")  # build a new model from scratch
# model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11l.pt")  # load a pretrained model (recommended for training)
# model = YOLO('runs/detect/train47/weights/best.pt') # load best.pt or last.pt of local model

# Use the model
model.train(data="/mnt/disk1/datasets/iNaturalist/Arthropods/FINAL_DATASET/Arthropoda.yaml", epochs=100, device=[0,1])  # train the model

# Evaluate model performance on the validation set
# metrics = model.val()
metrics = model.val(data="/mnt/disk1/datasets/iNaturalist/Arthropods/FINAL_DATASET/Arthropoda.yaml", split="test")

#path = model.export(format="onnx")