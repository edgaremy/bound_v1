import os
import os

import matplotlib.pyplot as plt
import plotly.express as px

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
dataset_folder = "/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/BD_307_cropped/dataset/test"
plot_class_repartition(dataset_folder)


def plot_class_hierarchy_repartition(dataset_folder):
    class_counts = {}
    class_hierarchies = {}
    
    # Iterate over the subfolders in the dataset folder
    for class_folder in os.listdir(dataset_folder):
        class_path = os.path.join(dataset_folder, class_folder)
        
        # Count the number of images in each class folder
        if os.path.isdir(class_path):
            class_counts[class_folder] = len(os.listdir(class_path))
            
            # Get the hierarchy of the class folder
            try:
                class_hierarchies[class_folder] = hierarchy.get_hierarchy_from_name(class_folder)
            except:
                class_hierarchies[class_folder] = ['', '', '', '', class_folder]
    
    # Plot the circular graph using plotly.express
    labels = class_counts.keys()
    counts = class_counts.values()
    
    fig = px.pie(names=labels, values=counts, title='Class Repartition')
    fig.show()

# Example Usage:
dataset_folder = "/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/BD_307_cropped/dataset/test"
plot_class_hierarchy_repartition(dataset_folder)