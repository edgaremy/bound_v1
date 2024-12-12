from ultralytics import YOLO
import csv
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import matplotlib
# matplotlib.use('agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

def get_metrics(label_lines, prediction_boxes, im_width, im_height, IoU_threshold=0.5):

    bbox_relative_sizes = []

    # Compute IoU
    combinations = np.zeros((len(prediction_boxes), len(label_lines)))
    for j in range(len(label_lines)):

        # Get the coordinates of the label bounding box
        line = label_lines[j].split(' ')
        label = int(line[0])
        x_center = float(line[1])
        y_center = float(line[2])
        width = float(line[3])
        height = float(line[4])

        # Compute the area of the bounding box (relative to the image size)
        bbox_relative_sizes.append(width * height)

        for i in range(len(prediction_boxes)):

            # Get the coordinates of the predicted bounding box
            pred_x1 = int(prediction_boxes[i][0])
            pred_y1 = int(prediction_boxes[i][1])
            pred_x2 = int(prediction_boxes[i][2])
            pred_y2 = int(prediction_boxes[i][3])

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
    IoUs = combinations[row_ind, col_ind]

    # Compute TP, FP, FN using IoU threshold (0.5 by default)
    TP = len(IoUs[IoUs > IoU_threshold])
    FP = len(prediction_boxes) - TP
    FN = len(label_lines) - TP

    return TP, FP, FN

def get_performance_metrics(model, path, split='test'):
    # Load model
    model = YOLO(model) # load best.pt or last.pt of local model      

    results = {}

    confidence_thresholds = np.linspace(0.01, 0.99, num=99)
    F1_scores = []

    for conf_threshold in confidence_thresholds:
        print(f"Confidence threshold: {conf_threshold}")

        TP, FP, FN = 0, 0, 0

        # Evaluate model performance on the validation set
        for img in tqdm(os.listdir(os.path.join(path, "images", split))):

            img_path = os.path.join(path, "images", split, img)
            ground_truth_label = os.path.join(path, "labels", split, img.split('.')[0] + '.txt')
            if not os.path.exists(ground_truth_label): # No label, nothing to measure
                continue

            results = model(img_path, save_txt=False, save=False, verbose=False, conf=conf_threshold)

            with open(ground_truth_label, 'r') as file:
                lines = file.readlines()
                boxes = results[0].boxes.xyxy.cpu().numpy()

                if len(lines) == 0: # No label, nothing to measure
                    continue
                else:
                    # Get original img size
                    im_width, im_height = results[0].boxes.orig_shape[0], results[0].boxes.orig_shape[1]
                    tp, fp, fn= get_metrics(lines, boxes, im_width, im_height)
                    TP += tp
                    FP += fp
                    FN += fn

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        F1_scores.append(F1)
    
    best_F1 = np.max(F1_scores)
    best_conf_threshold = confidence_thresholds[np.argmax(F1_scores)]
    print(f"Best confidence threshold: {best_conf_threshold}, Best F1 score: {best_F1}")

    # Plot the F1 score depending on confidence threshold
    plt.plot(confidence_thresholds, F1_scores)
    plt.xlabel('Confidence threshold')
    plt.ylabel('F1 score')
    # add best confidence threshold to the plot legend
    plt.legend([f"Best F1 score: {best_F1} at confidence threshold: {best_conf_threshold}"])
    # save the plot
    plt.savefig('arthropods_dataset_scripts/benchmark/F1_score_vs_IoU_threshold.png')


# Example usage:
model_path = "runs/detect/train45/weights/best.pt"
path = "/mnt/disk1/datasets/iNaturalist/Arthropods/FINAL_DATASET/"
get_performance_metrics(model_path, path, split='test')