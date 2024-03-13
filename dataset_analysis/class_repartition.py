import os
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

import get_hierarchy as hierarchy

def plot_class_repartition(dataset_folder):
    class_counts = {}
    
    # Iterate over the subfolders in the dataset folder
    for class_folder in os.listdir(dataset_folder):
        class_path = os.path.join(dataset_folder, class_folder)
        
        # Count the number of images in each class folder
        if os.path.isdir(class_path):
            class_counts[class_folder] = len(os.listdir(class_path))
    
    # Plot the circular graph
    labels = class_counts.keys()
    counts = class_counts.values()
    
    plt.pie(counts, labels=labels, autopct='%1.1f%%')
    plt.axis('equal')
    plt.show()

# Example Usage:
# dataset_folder = "/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/BD_307_cropped/dataset/test"
# plot_class_repartition(dataset_folder)


def plot_class_hierarchy_repartition(dataset_folders):
    class_counts = {}
    class_hierarchies = {}
    
    for dataset_folder in dataset_folders:

        # Iterate over the subfolders in the dataset folder
        for class_folder in os.listdir(dataset_folder):
            class_path = os.path.join(dataset_folder, class_folder)
            
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

    data = {
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
    fig = px.sunburst(df_, path=['class_', 'order', 'family', 'genus', 'species'], color='family', values='count')

    # Affichage du graphique
    fig.show()
    fig.write_html("./export.html")
    # fig.write_image("hierarchie_especes.svg")

# Example Usage:
# dataset_folders = ["/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/BD_307_cropped/dataset/test",
#                    "/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/BD_307_cropped/dataset/train",
#                    "/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/BD_307_cropped/dataset/val"]
dataset_folders = ["/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD_71_visited/classes"]
plot_class_hierarchy_repartition(dataset_folders)