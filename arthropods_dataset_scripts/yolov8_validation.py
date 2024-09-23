from ultralytics import YOLO
import csv
import os

def get_mean_iou(model):

    path = "/mnt/disk1/datasets/iNaturalist/Arthropods/dataset17/" #TODO use argument instead

    IoUs = [] # To evaluate the mean IoU

    for img in os.listdir(os.path.join(path, "images/test")):
        img_path = os.path.join(path, "images/test", img)
        results = model(img_path, save_txt=False, save=False, verbose=False)

        ground_truth_label = os.path.join(path, "labels/test" , img.split('.')[0] + '.txt')

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

                        IoUs.append(IoU)

                    break

    mean_iou = sum(IoUs) / len(IoUs)
    print("Mean IoU: ", mean_iou)
    return mean_iou

# Delete last val folder if it is empty
def delete_last_val_folder_if_empty(verbose=True):
    # find the last val folder
    path = "runs/detect/val"
    val_nb = 2
    last_found = True
    while last_found:
        val_nb += 1
        path_n = path + str(val_nb)
        if not os.path.exists(path_n):
            last_found = False
            val_nb -= 1
    
    path = path + str(val_nb)
    
    if os.path.exists(path) and os.path.isdir(path):
        if not os.listdir(path):
            os.rmdir(path)
    
    if verbose:
        print("Deleted empty folder: ", path)

def eval_yolov8(model_path, yaml_path, split='test'):
    # Load model
    model = YOLO(model_path) # load best.pt or last.pt of local model

    # Evaluate model performance on the validation set
    metrics = model.val(data=yaml_path, split=split, plots=False)
    delete_last_val_folder_if_empty()

    # Evaluate mean IoU
    mean_iou = get_mean_iou(model)

    # Evaluate F1-score
    f1_score = metrics.box.f1[0]

    return metrics.results_dict, mean_iou, f1_score

def save_metrics(model_paths, yaml_path, export_csv_path, split='test'):

    wave_number = 1
    # Write the metrics to a csv file
    with open(export_csv_path, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(["model_path", "wave", "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)", "fitness", "mean_IoU", "F1-score"])
        for model_path in model_paths:
            complete_model_path = "runs/detect/" + model_path + "/weights/best.pt"

            metrics, mean_iou, f1_score = eval_yolov8(complete_model_path, yaml_path, split)

            writer.writerow([model_path, wave_number, metrics["metrics/precision(B)"], metrics["metrics/recall(B)"], metrics["metrics/mAP50(B)"], metrics["metrics/mAP50-95(B)"], metrics["fitness"], mean_iou, f1_score])
            wave_number += 1


model_paths = ["train18", "train22", "train23", "train24", "train25", "train26", "train28", "train29", "train31", "train32", "train33", "train34", "train35", "train36", "train37", "train38"]
yaml_path = "/mnt/disk1/datasets/iNaturalist/Arthropods/dataset17/Arthropods17.yaml"
export_csv_path = "arthropods_dataset_scripts/test_metrics.csv"
save_metrics(model_paths, yaml_path, export_csv_path, split='test')