from ultralytics import YOLO
import ultralytics
import torch
import os

# Load model
model = YOLO('runs/detect/train16/weights/best.pt') # load best.pt or last.pt of local model

path = "/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/DG_evaluate_predictions/dataset/"

# Evaluate mean IoU:
IoUs = []

for img in os.listdir(os.path.join(path, "images/val")):
    img_path = os.path.join(path, "images/val", img)
    results = model(img_path, save_txt=False, save=False, verbose=False)

    ground_truth_label = os.path.join(path, "labels/val" , img.split('.')[0] + '.txt')

    if os.path.exists(ground_truth_label):
        with open(ground_truth_label, 'r') as file:
            lines = file.readlines()

            boxes = results[0].boxes.xyxy.cpu().numpy()
            for line in lines:

                if len(boxes) > 0:
                    # Get the coordinates of the predicted bounding box
                    pred_x1 = int(boxes[0][0])
                    pred_y1 = int(boxes[0][1])
                    pred_x2 = int(boxes[0][2])
                    pred_y2 = int(boxes[0][3])

                    # Get original img size
                    im_width, im_height = results[0].boxes.orig_shape[0], results[0].boxes.orig_shape[1]

                    line = line.split(' ')
                    label = int(line[0])
                    x_center = float(line[1])
                    y_center = float(line[2])
                    width = float(line[3])
                    height = float(line[4])

                    # Get the class name
                    class_name = model.names[label]

                    # Get the coordinates of the bounding box
                    x1 = int((x_center - width/2) * im_height)
                    y1 = int((y_center - height/2) * im_width)
                    x2 = int((x_center + width/2) * im_height)
                    y2 = int((y_center + height/2) * im_width)

                    print(x1, y1, x2, y2)
                    print(pred_x1, pred_y1, pred_x2, pred_y2)


                    # Compute the intersection over union
                    intersection = max(0, min(x2, pred_x2) - max(x1, pred_x1)) * max(0, min(y2, pred_y2) - max(y1, pred_y1))
                    area1 = (x2 - x1) * (y2 - y1)
                    area2 = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
                    union = area1 + area2 - intersection
                    IoU = intersection / union
                    
                    IoUs.append(IoU)

                break

print("Mean IoU: ", sum(IoUs) / len(IoUs))