import os
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

import get_hierarchy as hierarchy


def plot_class_hierarchy_repartition_from_taxon_id(dataset_folders):
    class_counts = {}
    class_hierarchies = {}

    for dataset_folder in dataset_folders:

        # Iterate over the image files in the dataset folder
        for file in os.listdir(dataset_folder):
            class_id = file.split('_')[0]
            if class_id not in class_counts:
                class_counts[class_id] = 1
            else:
                class_counts[class_id] += 1
            if class_id not in class_hierarchies:
                try:
                    class_hierarchies[class_id] = hierarchy.get_hierarchy_from_taxon_id(class_id)
                except:
                    print("ERROR, could not find hierarchy for class_id=", class_id)
                class_hierarchies[class_id].append(class_counts[class_id])
            else:
                class_hierarchies[class_id][5] = class_counts[class_id]

    data = {
        'phylum': ["Arthropoda" for class_folder in class_counts.keys()],
        'class_': [class_hierarchies[class_folder][0] for class_folder in class_counts.keys()],
        'order': [class_hierarchies[class_folder][1] for class_folder in class_counts.keys()],
        'family': [class_hierarchies[class_folder][2] for class_folder in class_counts.keys()],
        'genus': [class_hierarchies[class_folder][3] for class_folder in class_counts.keys()],
        'species': [class_hierarchies[class_folder][4] for class_folder in class_counts.keys()],
        'count': [class_hierarchies[class_folder][5] for class_folder in class_counts.keys()]
    }
        

    df_ = pd.DataFrame(data)

    df_.fillna('unknown', inplace=True)

    # Création du diagramme en treillis
    fig = px.sunburst(df_,
                      path=['phylum', 'class_', 'order', 'family'],
                      color='order', values='count',
                      template='presentation',
                      color_discrete_sequence=px.colors.qualitative.T10,
                      hover_data={'count': True, 'class_': False, 'order': False, 'family': False, 'genus': False, 'species': False})
    # fig = px.sunburst(df_, path=['class_', 'order', 'family', 'genus', 'species'], color='family', values='count')

    fig.update_layout(hoverlabel=dict(font_size=18, font_family="Rockwell"))
    fig.update_traces(hovertemplate=
                      '<b>%{label}</b><br><br>' +
                      'Total Images: %{value}<br>' +
                      'Taxon: %{id}<extra></extra>')
    fig.show()
    fig.write_html("./export.html")
    # fig.write_image("hierarchie_especes.svg")

# Example Usage:
dataset_folders = ["/mnt/disk1/datasets/iNaturalist/Arthropods/FINAL_DATASET/images/test"]#,
                #    "/mnt/disk1/datasets/iNaturalist/Arthropods/FINAL_DATASET/images/train",
                #    "/mnt/disk1/datasets/iNaturalist/Arthropods/FINAL_DATASET/images/val"]
plot_class_hierarchy_repartition_from_taxon_id(dataset_folders)