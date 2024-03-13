from ultralytics import YOLO
import numpy as np
import os

# Load a model
# model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8s-cls.pt')  # load a pretrained model (recommended for training)
model = YOLO('runs/classify/train5/weights/best.pt')

class_names = model.names

# # Use the model
# # model.train(data="/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/Cropped_BD_71_True/images", epochs=100, imgsz=640) # train the model
# # model.train(data="/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/BD_307_cropped/dataset", epochs=100, imgsz=640) # train the model
# metrics = model.val(data="/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/BD_307_cropped/dataset", classes = remaining_classes) # evaluate model performance on the validation set
# #results = model("/home/edgarremy/Documents/CODE/bound_v1/splitted_dataset/Task_Lepinoc/images/val/1d5d9f5767cfa18c8ad2594651f94753_0_2.jpg")  # predict on an image
# #path = model.export(format="onnx")


i = 0
remaining_classes = []
# conf_mat = np.zeros((len(remaining_classes), len(remaining_classes)))
conf_mat = np.zeros((306, 306))
for folder in os.listdir("/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/BD_307_cropped/dataset/test"):
    class_idx = list(class_names.keys())[list(class_names.values()).index(folder)]
    remaining_classes.append(class_idx)
    # print("Class index: ", class_idx)
    results = model(os.path.join("/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/BD_307_cropped/dataset/test/", folder),
                    classes = remaining_classes,
                    verbose=False)  # predict on a folder
    print("Predicted images of class: ", folder)
    for r in results:
        # print(r.probs.top1)
        conf_mat[class_idx, r.probs.top1] += 1

print("Total of classes encountered: ", len(remaining_classes))
# Print the results
# confusion_matrix = metrics.confusion_matrix.matrix
# Counting true positives, false positives, true negatives, and false negatives
true_positives = np.diag(conf_mat)
false_positives = np.sum(conf_mat, axis=1) - true_positives
false_negatives = np.sum(conf_mat, axis=0) - true_positives
true_negatives = np.sum(conf_mat) - true_positives - false_positives - false_negatives
# F1 Score
f1_score = 2 * true_positives / (2 * true_positives + false_positives + false_negatives)

# Keep only remaining classes:
true_positives = true_positives[remaining_classes]
false_positives = false_positives[remaining_classes]
false_negatives = false_negatives[remaining_classes]
true_negatives = true_negatives[remaining_classes]
f1_score = f1_score[remaining_classes]


# Print the results
# print("True Positives: ", true_positives)
# print("False Positives: ", false_positives)
# print("False Negatives: ", false_negatives)
# print("True Negatives: ", true_negatives)
# print("F1 Score: ", f1_score)
print("Mean F1 Score: ", np.mean(f1_score))
print("Top1 Accuracy: ", np.sum(true_positives) / np.sum(conf_mat))