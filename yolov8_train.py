from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
# model = YOLO('runs/detect/train16/weights/best.pt') # load best.pt or last.pt of local model


# Use the model
model.train(data="/mnt/disk1/datasets/Lepinoc_2022/splitted_dataset/Task_Lepinoc/Task Lepinoc.yaml", epochs=100)  # train the model
metrics = model.val()  # evaluate model performance on the validation set

#path = model.export(format="onnx")