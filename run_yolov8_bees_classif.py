from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8s-cls.pt')  # load a pretrained model (recommended for training)
# model = YOLO('runs/classify/train2/weights/best.pt')


# Use the model
model.train(data="/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/Cropped_BD_71_True/images", epochs=100, imgsz=640) # train the model
# metrics = model.val(batch=1)  # evaluate model performance on the validation set
#results = model("/home/edgarremy/Documents/CODE/bound_v1/splitted_dataset/Task_Lepinoc/images/val/1d5d9f5767cfa18c8ad2594651f94753_0_2.jpg")  # predict on an image
#path = model.export(format="onnx")


# # Display the image with bounding boxes
# image_path = "/home/edgarremy/Documents/CODE/bound_v1/splitted_dataset/Task_Lepinoc/images/val/1d5d9f5767cfa18c8ad2594651f94753_0_2.jpg"
# image = Image.open(image_path)
# image_np = np.array(image)

# plt.imshow(image_np)
# plt.axis('off')

# for result in results.xyxy[0]:
#     bbox = result[0:4]
#     label = result[5]
#     plt.gca().add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], fill=False, edgecolor='r', linewidth=2))
#     plt.text(bbox[0], bbox[1]-10, label, color='r')

# plt.show()

