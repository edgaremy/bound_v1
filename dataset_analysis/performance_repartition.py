import os
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from ultralytics import YOLO
import numpy as np
import os

import get_hierarchy as hierarchy
import sub_confusion_matrix as scm

def make_ordered_unique(list_):
    output = []
    indexes = []
    for i in range(len(list_)):
        if list_[i] not in output:
            output.append(list_[i])
            indexes.append(i)
    return np.array(output), np.array(indexes)

# Associate each element of current_hierarchy to an index of the upper hierarchy
def get_upper_hierarchy_index(current_hierarchy, upper_hierarchies):
    # Associate each unique upper_hierarchy to an index
    upper_hierarchy_unique, upper_hierarchy_indexes = make_ordered_unique(upper_hierarchies)
    upper_indexes = []
    # For each current, get the index of the upper_hierarchy
    for i in range(len(current_hierarchy)):
        upper_name = upper_hierarchies[i]
        upper_index = np.where(upper_hierarchy_unique == upper_name)[0]
        upper_indexes.append(upper_index[0])
    return np.array(upper_indexes)
        

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


    class_ = np.array([class_hierarchies[class_folder][0] for class_folder in class_counts.keys()])
    order = np.array([class_hierarchies[class_folder][1] for class_folder in class_counts.keys()])
    family = np.array([class_hierarchies[class_folder][2] for class_folder in class_counts.keys()])
    genus = np.array([class_hierarchies[class_folder][3] for class_folder in class_counts.keys()])
    species = np.array([class_hierarchies[class_folder][4] for class_folder in class_counts.keys()])
    count = np.array([class_hierarchies[class_folder][5] for class_folder in class_counts.keys()])

    class_unique, class_indexes = make_ordered_unique(class_)
    order_unique, order_indexes = make_ordered_unique(order)
    family_unique, family_indexes = make_ordered_unique(family)
    genus_unique, genus_indexes = make_ordered_unique(genus)

    # Per Species F1 Score:
    f1_score, _, _, _, _ = scm.confusion_matrix_stats(conf_mat)
    f1_score_species = f1_score[remaining_classes] # Keep only remaining classes
    print('Species F1 Score: ', np.mean(f1_score_species))

    # Per Genus F1 Score:
    species_genus_idx = get_upper_hierarchy_index(species, genus)
    conf_mat_genus = scm.merge_confusion_matrix(conf_mat, species_genus_idx)
    f1_score, _, _, _, _ = scm.confusion_matrix_stats(conf_mat_genus)
    f1_score_genus = f1_score[:] # Keep only remaining classes
    counts_genus = np.zeros(len(genus_unique))
    for i in range(len(species_genus_idx)):
        counts_genus[species_genus_idx[i]] += count[i]
    print('Genus F1 Score: ', np.mean(f1_score_genus))

    # Per Family F1 Score:
    species_family_idx = get_upper_hierarchy_index(species, family)
    conf_mat_family = scm.merge_confusion_matrix(conf_mat, species_family_idx)
    f1_score, _, _, _, _ = scm.confusion_matrix_stats(conf_mat_family)
    f1_score_family = f1_score[:] # Keep only remaining classes
    counts_family = np.zeros(len(family_unique))
    for i in range(len(species_family_idx)):
        counts_family[species_family_idx[i]] += count[i]
    print('Family F1 Score: ', np.mean(f1_score_family))

    # Counting for remaining order and class_:
    species_order_idx = get_upper_hierarchy_index(species, order)
    counts_order = np.zeros(len(order_unique))
    for i in range(len(species_order_idx)):
        counts_order[species_order_idx[i]] += count[i]

    species_class_idx = get_upper_hierarchy_index(species, class_)
    counts_class = np.zeros(len(class_unique))
    for i in range(len(species_class_idx)):
        counts_class[species_class_idx[i]] += count[i]

    data = dict(
        names = np.concatenate((class_unique,order_unique,family_unique,genus_unique,species), axis=None),
        parents = np.concatenate((np.array([''] * len(class_unique)), class_[order_indexes], order[family_indexes], family[genus_indexes], genus), axis=None),
        values = np.concatenate((counts_class, counts_order, counts_family, counts_genus, count), axis=None),
        f1_scores = np.concatenate((np.array([1] * len(class_unique)), np.array([1] * len(order_unique)), f1_score_family, f1_score_genus, f1_score_species), axis=None)
    )

# ['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',
#              'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',
#              'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',
#              'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',
#              'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',
#              'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',
#              'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',
#              'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl',
#              'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn',
#              'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu',
#              'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar',
#              'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',
#              'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',
#              'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr',
#              'ylorrd']

    fig = px.sunburst(data,
                      names = 'names',
                      parents = 'parents',
                      values = 'values',
                      color = 'f1_scores',
                      branchvalues = 'total',
                      color_continuous_scale='fall_r')
    # fig.update_layout(uniformtext=dict(minsize=13, mode='hide'))
    # Affichage du graphique 
    fig.show()
    fig.write_html("./export.html")
    # fig.write_image("hierarchie_especes.svg")

# Example Usage:
model_path = 'runs/classify/train5/weights/best.pt'
dataset_folder = "/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/BD_307_cropped/dataset/test"
plot_class_hierarchy_performance(dataset_folder, model_path)