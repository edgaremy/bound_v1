import os
os.environ['YOLO_VERBOSE'] = 'false'
from ultralytics import YOLO
import csv
import glob

def get_mean_iou(model, subtype, path):

    IoUs = []  # To evaluate the mean IoU

    images = glob.glob(os.path.join(path, "images/test", f"{subtype}_*"))
    for img_path in images:

        img = os.path.basename(img_path)
        results = model(img_path, save_txt=False, save=False, verbose=False)

        ground_truth_label = os.path.join(path, "labels/test", img.split('.')[0] + '.txt')

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

    mean_iou = sum(IoUs) / len(IoUs) if IoUs else 0
    # print(f"Mean IoU for subtype {subtype}: ", mean_iou)
    return mean_iou

def delete_last_val_folder_if_empty(verbose=False):
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

def eval_yolov8(model, yaml_path, subtype, split='test'):

    # Check if there is at least one image with label for the subtype:
    label_path = os.path.join(os.path.dirname(yaml_path), "labels", split)
    # is there at least one label starting with subtype_?
    if not any([label.startswith(subtype + "_") for label in os.listdir(label_path)]):
        print(f"No label found for subtype {subtype} in {label_path}")
        return None, None, None

    # Remove any cache that disturb the evaluation:
    cache_dir = os.path.join(os.path.dirname(yaml_path), "labels", split + ".cache")
    if os.path.isfile(cache_dir):
        os.remove(cache_dir)
        # print("\nRemoved cache file: ", cache_dir, "\n\n")
    
    # Evaluate model performance on the validation set
    metrics = model.val(data=yaml_path, split=split, plots=False, verbose=False, workers=0)
    delete_last_val_folder_if_empty()

    path = "/mnt/disk1/datasets/iNaturalist/Arthropods/dataset17/"
    mean_iou = get_mean_iou(model, subtype, path)

    if len(metrics.box.f1) == 0:
        f1_score = 0
    else:
        f1_score = metrics.box.f1[0]

    return metrics.results_dict, mean_iou, f1_score

# Saves the metrics to a csv file, for each model_path applied to each yaml file
def save_metrics(model_paths, yaml_list, export_csv_path):
    wave_number = 0

    with open(export_csv_path, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(["model_path", "wave", "subtype", "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)", "fitness", "mean_IoU", "F1-score"])

        for model_path in model_paths:
            print(f"\nPROCESSING MODEL OF WAVE {wave_number}")
            complete_model_path = "runs/detect/" + model_path + "/weights/best.pt"
            model = YOLO(complete_model_path, verbose=False)  # Load model once per model_path

            for yaml_path in yaml_list:
                subtype = yaml_path.split('/')[-1].split('_')[0]
                # print(f"Processing model {model_path} for subtype {subtype}")
                
                metrics, mean_iou, f1_score = eval_yolov8(model, yaml_path, subtype)
                if metrics is None:
                    continue
                writer.writerow([model_path, wave_number, subtype, metrics["metrics/precision(B)"], metrics["metrics/recall(B)"], metrics["metrics/mAP50(B)"], metrics["metrics/mAP50-95(B)"], metrics["fitness"], mean_iou, f1_score])

            wave_number += 1


####### Example of use: #######

model_paths = ["train16", "train18", "train22", "train23", "train24", "train25", "train26", "train28", "train29", "train31", "train32", "train33", "train34", "train35", "train36", "train37", "train38"]
export_csv_path = "arthropods_dataset_scripts/test_metrics_families.csv"

# List all the family's subsets:
base_directory = "/mnt/disk1/datasets/iNaturalist/Arthropods/dataset17/"
yaml_list = os.listdir(base_directory)

# Keep only the yaml files ending with "_subset.yaml"
yaml_list = [yaml for yaml in yaml_list if yaml.endswith("_subset.yaml")]
yaml_list = [os.path.join(base_directory, yaml) for yaml in yaml_list]

# For each yaml file, saves metrics to the csv file
save_metrics(model_paths, yaml_list, export_csv_path)