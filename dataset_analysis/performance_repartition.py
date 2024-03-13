import os
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from ultralytics import YOLO
import numpy as np
import os

import get_hierarchy as hierarchy

def plot_class_hierarchy_performance(dataset_folder, model_path):
    # Load a model
    model = YOLO(model_path)
    class_names = model.names
    remaining_classes = []
    conf_mat = np.zeros((306, 306))

    class_counts = {}
    class_hierarchies = {}

    i = 0
    # Iterate over the subfolders in the dataset folder
    for class_folder in os.listdir(dataset_folder):
        class_path = os.path.join(dataset_folder, class_folder)
        

        # PREDICT on images of the class folder and MAKE confusion matrix
        class_idx = list(class_names.keys())[list(class_names.values()).index(class_folder)]
        remaining_classes.append(class_idx)
        print("Predicting images of class: ", class_folder)
        results = model(os.path.join(dataset_folder, class_folder),
                        classes = remaining_classes,
                        verbose=False)  # predict on a folder
        for r in results:
            # print(r.probs.top1)
            conf_mat[class_idx, r.probs.top1] += 1


        # Count the number of images in each class folder
        if os.path.isdir(class_path):
            if class_folder not in class_counts:
                class_counts[class_folder] = len(os.listdir(class_path))
            else:
                class_counts[class_folder] += len(os.listdir(class_path))
            
            # Get the hierarchy of the class folder
            if class_folder not in class_hierarchies:
                try:
                    class_hierarchies[class_folder] = hierarchy.get_hierarchy_from_name(class_folder)
                except:
                    genus = class_folder.split(' ')[0]
                    class_hierarchies[class_folder] = [None, None, None, genus, class_folder]
                class_hierarchies[class_folder].append(class_counts[class_folder])
            else:
                class_hierarchies[class_folder][5] = class_counts[class_folder]
        
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

    print('Global F1 Score: ', np.mean(f1_score))


    # PLOT graph of results
    data = {
        'class_': [class_hierarchies[class_folder][0] for class_folder in class_counts.keys()],
        'order': [class_hierarchies[class_folder][1] for class_folder in class_counts.keys()],
        'family': [class_hierarchies[class_folder][2] for class_folder in class_counts.keys()],
        'genus': [class_hierarchies[class_folder][3] for class_folder in class_counts.keys()],
        'species': [class_hierarchies[class_folder][4] for class_folder in class_counts.keys()],
        'count': [class_hierarchies[class_folder][5] for class_folder in class_counts.keys()],
        'F1 Score': f1_score
    }
        

    df_ = pd.DataFrame(data)

    df_.fillna('unknown', inplace=True)

    # Création du diagramme en treillis
    fig = px.sunburst(df_, path=['class_', 'order', 'family', 'genus', 'species'], color='F1 Score', values='count')

    # Affichage du graphique
    fig.show()
    fig.write_html("./export.html")
    # fig.write_image("hierarchie_especes.svg")

# Example Usage:
model_path = 'runs/classify/train5/weights/best.pt'
dataset_folder = "/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/BD_307_cropped/dataset/test"
plot_class_hierarchy_performance(dataset_folder, model_path)