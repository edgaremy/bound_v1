import os
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from ultralytics import YOLO
import numpy as np
import os

def create_hierarchy_matrices(csv_file):
    hierarchy = pd.read_csv(csv_file)

    specie = list(hierarchy["specie"].unique())
    nb_specie = len(specie)

    genus = list(hierarchy["genus"].unique())
    nb_genus = len(genus)

    family = list(hierarchy["family"].unique())
    nb_family = len(family)

    order = list(hierarchy["order"].unique())
    nb_order = len(order)

    class_ = list(hierarchy["class"].unique())
    nb_class = len(class_)

    species_to_genus = np.zeros((nb_genus, nb_specie))
    genus_to_family = np.zeros((nb_family, nb_genus))
    family_to_order = np.zeros((nb_order, nb_family))
    order_to_class = np.zeros((nb_class, nb_order))
    for i in range(nb_specie):
        # species -> genus
        genus_species = hierarchy.at[i, "genus"]
        ind_genus = genus.index(genus_species)
        species_to_genus[ind_genus, i] = 1


        # genus -> family
        family_species = hierarchy.at[i, "family"]
        ind_family = family.index(family_species)
        genus_to_family[ind_family, ind_genus] = 1

        # family -> order
        order_species = hierarchy.at[i, "order"]
        ind_order = order.index(order_species)
        family_to_order[ind_order, ind_family] = 1

        # order -> class
        class_species = hierarchy.at[i, "class"]
        ind_class = class_.index(class_species)
        order_to_class[ind_class, ind_order] = 1
    
    # Make dictionnary with the matrices
    hierarchy_matrices = {
        "species_to_genus": species_to_genus,
        "genus_to_family": genus_to_family,
        "family_to_order": family_to_order,
        "order_to_class": order_to_class
    }

    return hierarchy_matrices, specie, genus, family, order, class_

def make_hierarchy_confusion_matrices(model_path, dataset_folder, hierarchy_csv_file):

    hierarchy_matrices, specie, genus, family, order, class_ = create_hierarchy_matrices(hierarchy_csv_file)
    nb_specie = len(specie)
    nb_genus = len(genus)
    nb_family = len(family)
    nb_order = len(order)
    nb_class = len(class_)
    conf_mat_species = np.zeros((nb_specie, nb_specie))
    conf_mat_genus = np.zeros((nb_genus, nb_genus))
    conf_mat_family = np.zeros((nb_family, nb_family))
    conf_mat_order = np.zeros((nb_order, nb_order))
    conf_mat_class = np.zeros((nb_class, nb_class))

    hierarchy = pd.read_csv(hierarchy_csv_file)
    encountered_species = []
    specie_count = np.zeros(nb_specie)

    # Loading the model
    class_folder = os.listdir(dataset_folder)[0]
    class_folder = os.path.join(dataset_folder, class_folder)
    model = YOLO(model_path)
    results = model(os.path.join(dataset_folder, class_folder), verbose=False) # predict on a folder

    # Iterate over the subfolders in the dataset folder
    for class_folder in os.listdir(dataset_folder):

        class_path = os.path.join(dataset_folder, class_folder)
        # if class_folder not in specie_count:
        #     specie_count[class_folder] = len(os.listdir(class_path))
        # else:
        #     specie_count[class_folder] += len(os.listdir(class_path))
        specie_count[specie.index(class_folder)] = len(os.listdir(class_path))

        specie_idx = specie.index(class_folder)
        if specie_idx not in encountered_species:
            encountered_species.append(specie_idx)
        results = model(class_path, verbose=False)  # predict on a folder

        for r in results:
            probabilities = r.probs.data.cpu().numpy()
            # species conf_mat
            conf_mat_species[specie_idx, probabilities.argmax()] += 1

            # genus conf_mat
            genus_name = hierarchy.at[specie_idx, "genus"]
            genus_idx = genus.index(genus_name)
            genus_probs = hierarchy_matrices["species_to_genus"] @ probabilities
            conf_mat_genus[genus_idx, genus_probs.argmax()] += 1

            # family conf_mat
            family_name = hierarchy.at[specie_idx, "family"]
            family_idx = family.index(family_name)
            family_probs = hierarchy_matrices["genus_to_family"] @ genus_probs
            conf_mat_family[family_idx, family_probs.argmax()] += 1

            # order conf_mat
            order_name = hierarchy.at[specie_idx, "order"]
            order_idx = order.index(order_name)
            order_probs = hierarchy_matrices["family_to_order"] @ family_probs
            conf_mat_order[order_idx, order_probs.argmax()] += 1

            # class conf_mat
            class_name = hierarchy.at[specie_idx, "class"]
            class_idx = class_.index(class_name)
            class_probs = hierarchy_matrices["order_to_class"] @ order_probs
            conf_mat_class[class_idx, class_probs.argmax()] += 1

    # Make dictionnary with the confusion matrices
    hierarchy_conf_mat = {
        "species": conf_mat_species,
        "genus": conf_mat_genus,
        "family": conf_mat_family,
        "order": conf_mat_order,
        "class": conf_mat_class
    }
    # Setting to zeros
    # count = np.array([specie_count[specie[i]] for i in range(nb_specie)])
    return hierarchy_conf_mat, specie, genus, family, order, class_, encountered_species, specie_count

def plot_hierarchy_confusion_matrices(hierarchy_conf_mat, specie, genus, family, order, class_):
    fig = go.Figure(data=go.Heatmap(
        z=hierarchy_conf_mat["species"],
        x=specie,
        y=specie,
        colorscale='Viridis'))

    fig.update_layout(
        title='Species confusion matrix',
        xaxis_nticks=36)

    fig.show()

    fig = go.Figure(data=go.Heatmap(
        z=hierarchy_conf_mat["genus"],
        x=genus,
        y=genus,
        colorscale='Viridis'))

    fig.update_layout(
        title='Genus confusion matrix',
        xaxis_nticks=36)

    fig.show()

    fig = go.Figure(data=go.Heatmap(
        z=hierarchy_conf_mat["family"],
        x=family,
        y=family,
        colorscale='Viridis'))

    fig.update_layout(
        title='Family confusion matrix',
        xaxis_nticks=36)

    fig.show()

    fig = go.Figure(data=go.Heatmap(
        z=hierarchy_conf_mat["order"],
        x=order,
        y=order,
        colorscale='Viridis'))

    fig.update_layout(
        title='Order confusion matrix',
        xaxis_nticks=36)

    fig.show()

    fig = go.Figure(data=go.Heatmap(
        z=hierarchy_conf_mat["class"],
        x=class_,
        y=class_,
        colorscale='Viridis'))

    fig.update_layout(
        title='Class confusion matrix',
        xaxis_nticks=36)
    
    fig.show()

def confusion_matrix_stats(conf_mat):
    # Counting true positives, false positives, true negatives, and false negatives
    true_positives = np.diag(conf_mat)
    false_positives = np.sum(conf_mat, axis=1) - true_positives
    false_negatives = np.sum(conf_mat, axis=0) - true_positives
    true_negatives = np.sum(conf_mat) - true_positives - false_positives - false_negatives
    # F1 Score
    f1_score = np.where((true_positives + false_positives + false_negatives) > 0, 2 * true_positives / (2 * true_positives + false_positives + false_negatives), 0)
    return f1_score, true_positives, false_positives, false_negatives, true_negatives

def compute_hierarchy_perfs(hierarchy_conf_mat, encountered_species):
    # Per Species F1 Score:
    f1_score, _, _, _, _ = confusion_matrix_stats(hierarchy_conf_mat["species"])
    f1_score_species = f1_score[:] # Keep only remaining classes
    print('Species F1 Score: ', np.mean(f1_score_species[encountered_species]))

    # Per Genus F1 Score:
    f1_score, _, _, _, _ = confusion_matrix_stats(hierarchy_conf_mat["genus"])
    f1_score_genus = f1_score[:] # Keep only remaining classes
    print('Genus F1 Score: ', np.mean(f1_score_genus))

    # Per Family F1 Score:
    f1_score, _, _, _, _ = confusion_matrix_stats(hierarchy_conf_mat["family"])
    f1_score_family = f1_score[:] # Keep only remaining classes
    print('Family F1 Score: ', np.mean(f1_score_family))

    # Per Order F1 Score:
    f1_score, _, _, _, _ = confusion_matrix_stats(hierarchy_conf_mat["order"])
    f1_score_order = f1_score[:] # Keep only remaining classes
    print('Order F1 Score: ', np.mean(f1_score_order))

    # Per Class F1 Score:
    f1_score, _, _, _, _ = confusion_matrix_stats(hierarchy_conf_mat["class"])
    f1_score_class = f1_score[:] # Keep only remaining classes
    print('Class F1 Score: ', np.mean(f1_score_class))

    return f1_score_species, f1_score_genus, f1_score_family, f1_score_order, f1_score_class



# Example usage:
model_path = 'runs/classify/train5/weights/best.pt'
dataset_folder = "/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/BD_307_cropped/dataset/test"
hierarchy_csv = "dataset_analysis/306_hierarchy.csv"
hierarchy_conf_mat, specie, genus, family, order, class_, encountered_species, count = make_hierarchy_confusion_matrices(model_path, dataset_folder, hierarchy_csv)
f1_score_species, f1_score_genus, f1_score_family, f1_score_order, f1_score_class = compute_hierarchy_perfs(hierarchy_conf_mat, encountered_species)
# plot_hierarchy_confusion_matrices(hierarchy_conf_mat, specie, genus, family, order, class_)


hierarchy = pd.read_csv(hierarchy_csv)

specie_genus_parent = []
genus_count = np.zeros(len(genus))
for i in range(len(specie)):
    specie_genus_parent.append(hierarchy.at[i, "genus"])
    genus_count[genus.index(hierarchy.at[i, "genus"])] += count[i]

genus_family_parent = []
family_count = np.zeros(len(family))
for i in range(len(genus)):
    line_idx = hierarchy[hierarchy["genus"] == genus[i]].index[0] # find first line with correct genus
    genus_family_parent.append(hierarchy.at[line_idx, "family"])
    family_count[family.index(hierarchy.at[line_idx, "family"])] += genus_count[i]

family_order_parent = []
order_count = np.zeros(len(order))
for i in range(len(family)):
    line_idx = hierarchy[hierarchy["family"] == family[i]].index[0] # find first line with correct family
    family_order_parent.append(hierarchy.at[line_idx, "order"])
    order_count[order.index(hierarchy.at[line_idx, "order"])] += family_count[i]

order_class_parent = []
class_count = np.zeros(len(class_))
for i in range(len(order)):
    line_idx = hierarchy[hierarchy["order"] == order[i]].index[0] # find first line with correct order
    order_class_parent.append(hierarchy.at[line_idx, "class"])
    class_count[class_.index(hierarchy.at[line_idx, "class"])] += order_count[i]

data = {
        'Name' : np.concatenate((class_,order,family,genus,specie), axis=None),
        'Parent' : np.concatenate((np.array([''] * len(class_)), order_class_parent, family_order_parent, genus_family_parent, specie_genus_parent), axis=None),
        'Count' : np.concatenate((class_count, order_count, family_count, genus_count, count), axis=None),
        'F1-score' : np.concatenate((f1_score_class, f1_score_order, f1_score_family, f1_score_genus, f1_score_species), axis=None)
}

fig = px.sunburst(data,
                    names = 'Name',
                    parents = 'Parent',
                    values = 'Count',
                    color = 'F1-score',
                    branchvalues = 'total',
                    color_continuous_scale='fall_r',
                    # hover_name = 'Name',
                    # hover_data={'Name': False, 'Parent': False, 'Count': True, 'F1-score': ':.2f'},
                    template='presentation'
                    )
fig.update_layout(hoverlabel=dict(font_size=18, font_family="Rockwell"))
fig.update_traces(hovertemplate=
                      '<b>%{label}</b><br><br>' +
                      'Total Images = %{value}<br>' +
                      'F1-score = %{color:.2f}<br>')

# Affichage du graphique 
fig.show()
fig.write_html("./export.html")
# fig.write_image("hierarchie_especes.svg")