from ultralytics import YOLO
import os
import numpy as np
from scipy.optimize import linear_sum_assignment

def get_mean_iou(model):

    path = "/home/eremy/datasets/iNaturalist/Arthropods/dataset17" #TODO use argument instead

    IoUs = [] # To evaluate the mean IoU

    for img in os.listdir(os.path.join(path, "images/test")):
        img_path = os.path.join(path, "images/test", img)
        results = model(img_path, save_txt=False, save=False, verbose=False)

        ground_truth_label = os.path.join(path, "labels/test" , img.split('.')[0] + '.txt')

        if os.path.exists(ground_truth_label):
            with open(ground_truth_label, 'r') as file:
                lines = file.readlines()
                boxes = results[0].boxes.xyxy.cpu().numpy()

                combinations = np.zeros((len(boxes), len(lines)))

                if len(lines) == 0: # If there are no ground truth boxes, no IoU is measured
                    break
                if len(boxes) == 0 and len(lines) > 0: # If there are no predicted boxes, the IoU is 0
                    IoUs.append(0)
                    break

                for j in range(len(lines)):
                    for i in range(len(boxes)):

                        # Get the coordinates of the predicted bounding box
                        pred_x1 = int(boxes[i][0])
                        pred_y1 = int(boxes[i][1])
                        pred_x2 = int(boxes[i][2])
                        pred_y2 = int(boxes[i][3])

                        # Get original img size
                        im_width, im_height = results[0].boxes.orig_shape[0], results[0].boxes.orig_shape[1]

                        line = lines[j].split(' ')
                        label = int(line[0])
                        x_center = float(line[1])
                        y_center = float(line[2])
                        width = float(line[3])
                        height = float(line[4])

                        # Get the class name
                        # class_name = model.names[label]

                        # Get the coordinates of the bounding box
                        x1 = int((x_center - width/2) * im_height)
                        y1 = int((y_center - height/2) * im_width)
                        x2 = int((x_center + width/2) * im_height)
                        y2 = int((y_center + height/2) * im_width)

                        # Compute the intersection over union
                        intersection = max(0, min(x2, pred_x2) - max(x1, pred_x1)) * max(0, min(y2, pred_y2) - max(y1, pred_y1))
                        area1 = (x2 - x1) * (y2 - y1)
                        area2 = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
                        union = area1 + area2 - intersection
                        IoU = intersection / union

                        combinations[i, j] = IoU

                
                row_ind, col_ind = linear_sum_assignment(-combinations)
                IoU = combinations[row_ind, col_ind].sum() / len(lines)
                IoUs.append(IoU)


    mean_iou = sum(IoUs) / len(IoUs) if IoUs else 0
    print("Mean IoU: ", mean_iou)
    return mean_iou

# Test the function
model = YOLO("runs/detect/train40/weights/best.pt")
get_mean_iou(model)
