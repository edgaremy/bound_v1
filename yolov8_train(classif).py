from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8s-cls.pt')  # load a pretrained model (recommended for training)
# model = YOLO('runs/classify/train2/weights/best.pt')


# Use the model
model.train(data="/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/Cropped_BD_71_True/images", epochs=100, imgsz=640) # train the model
metrics = model.val(data=None)  # evaluate model performance on the validation set
#results = model("/home/edgarremy/Documents/CODE/bound_v1/splitted_dataset/Task_Lepinoc/images/val/1d5d9f5767cfa18c8ad2594651f94753_0_2.jpg")  # predict on an image
#path = model.export(format="onnx")
