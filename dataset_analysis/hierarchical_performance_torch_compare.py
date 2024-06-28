import os
import pandas as pd
import numpy as np
import glob

import torch
os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"]="1" # second gpu
import keras
keras.backend.set_image_data_format("channels_first")

torch.cuda.empty_cache()

import timm
import torch.nn as nn

from timm.data.dataset import ImageDataset
from timm.data.loader import create_loader

from torchvision.io import read_image
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def compute_hierarchy_perfs(model_path):

    IMG_SIZE = 224
    nb_classes = 306
    model = timm.create_model("resnet50.a1_in1k", pretrained=True, num_classes=nb_classes)
    # model.load_state_dict(torch.load("./model_best.pth.tar")['model_state_dict'])
    model.load_state_dict(torch.load(model_path)['state_dict'])
    model = nn.Sequential(model, nn.Softmax(dim=1)) # adding softmax activation at the end of the model
    model.eval()


    class CustomImageTestDataset(Dataset):
        def __init__(self, test_dir, class_folders_dir, transform=None, target_transform=None):
            all_classes_dir = glob.glob(os.path.join(class_folders_dir, "*/*"))
            all_classes_labels = [img.split("/")[-2] for img in all_classes_dir]

            self.img_dir = glob.glob(os.path.join(test_dir, "*/*"))
            labels = [img.split("/")[-2] for img in self.img_dir]
            self.class_names = sorted(set(all_classes_labels))
            self.nb_classes = len(self.class_names)
            self.img_labels = [self.class_names.index(l) for l in labels]
            self.transform = transform
            self.target_transform = target_transform

        def __len__(self):
            return len(self.img_labels)

        def __getitem__(self, idx):
            img_path = self.img_dir[idx]
            image = read_image(img_path)
            label = self.img_labels[idx]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            # label = np.expand_dims(label,-1)
            # make label hot one vector
            label = torch.nn.functional.one_hot(torch.as_tensor(label), num_classes=self.nb_classes)
            return image, label

    normalize = v2.Compose([
        v2.Resize((IMG_SIZE, IMG_SIZE), interpolation=v2.InterpolationMode.BICUBIC),
        v2.ToDtype(torch.float32),
        # v2.Normalize(mean=[0], std=[255]),
        # v2.ToTensor()
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    BATCH_SIZE = 128
    test_dir = "/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/BD_307_cropped/dataset/test"
    class_dir = "/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/BD_307_cropped/dataset/train"
    test_data = CustomImageTestDataset(test_dir, class_dir, normalize)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    nb_classes = test_data.nb_classes

    dataset = ImageDataset("/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/BD_307_cropped/dataset/test", class_map="./torch_benchmark/class_to_idx_mapping.txt")#, transform=normalize)
    test_loader = create_loader(dataset, (3, 224, 224), batch_size=BATCH_SIZE, is_training=False, num_workers=8)

    labels = []
    prediction = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    predictions = np.zeros((len(test_data), nb_classes), dtype=float)
    # for i, (image_batch, label_batch) in enumerate(tqdm(test_dataloader)):
    for i, (image_batch, label_batch) in enumerate(test_loader):
        pred = model(image_batch).cpu().detach().numpy()
        y = 0
        for label in label_batch:
            # label_idx = label.argmax() # if label is one hot encoded
            label_idx = label.cpu() # if label is an index
            labels.append(label_idx)
            predictions[y+BATCH_SIZE*i,:] = pred[y,:]
            y += 1
        pred_idx = pred.argmax(axis=1)
        prediction += pred_idx.tolist()
    labels = np.array([int(label) for label in labels], dtype=int)
    prediction = np.array(prediction, dtype=int)
    print(sum(prediction == labels), "/", len(labels))

    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def create_hierarchy_matrices(csv_file):
        hierarchy = pd.read_csv(csv_file)

        # specie = list(hierarchy["specie"].unique())
        # specie = sorted(list(hierarchy["specie"].unique()))
        specie = test_data.class_names
        nb_specie = len(specie)

        genus = sorted(list(hierarchy["genus"].unique()))
        nb_genus = len(genus)

        family = sorted(list(hierarchy["family"].unique()))
        nb_family = len(family)

        order = sorted(list(hierarchy["order"].unique()))
        nb_order = len(order)

        class_ = sorted(list(hierarchy["class"].unique()))
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

    def make_hierarchy_confusion_matrices(dataset_folder, hierarchy_csv_file):

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

        # Iterate over the subfolders in the dataset folder
        for class_folder in os.listdir(dataset_folder):

            class_path = os.path.join(dataset_folder, class_folder)
            specie_count[specie.index(class_folder)] = len(os.listdir(class_path))

            specie_idx = specie.index(class_folder)
            if specie_idx not in encountered_species:
                encountered_species.append(specie_idx)
            # results = model(class_path, verbose=False)  # predict on a folder

        for pred in range(predictions.shape[0]):
            probabilities = predictions[pred,:]
            # probabilities = softmax(probabilities) # if no activation function at the end of the model
            specie_idx = labels[pred]
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
        mean_species = np.mean(f1_score_species[encountered_species])
        print('Species F1 Score: ', mean_species)

        # Per Genus F1 Score:
        f1_score, _, _, _, _ = confusion_matrix_stats(hierarchy_conf_mat["genus"])
        f1_score_genus = f1_score[:] # Keep only remaining classes
        mean_genus = np.mean(f1_score_genus)
        print('Genus F1 Score: ', mean_genus)

        # Per Family F1 Score:
        f1_score, _, _, _, _ = confusion_matrix_stats(hierarchy_conf_mat["family"])
        f1_score_family = f1_score[:] # Keep only remaining classes
        mean_family = np.mean(f1_score_family)
        print('Family F1 Score: ', mean_family)

        # Per Order F1 Score:
        f1_score, _, _, _, _ = confusion_matrix_stats(hierarchy_conf_mat["order"])
        f1_score_order = f1_score[:] # Keep only remaining classes
        mean_order = np.mean(f1_score_order)
        print('Order F1 Score: ', mean_order)

        # Per Class F1 Score:
        f1_score, _, _, _, _ = confusion_matrix_stats(hierarchy_conf_mat["class"])
        f1_score_class = f1_score[:] # Keep only remaining classes
        mean_class = np.mean(f1_score_class)
        print('Class F1 Score: ', mean_class)

        return mean_species, mean_genus, mean_family, mean_order, mean_class



    # Example usage:
    model_path = None
    dataset_folder = "/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/BD_307_cropped/dataset/test"
    hierarchy_csv = "dataset_analysis/306_hierarchy.csv"
    hierarchy_conf_mat, specie, genus, family, order, class_, encountered_species, count = make_hierarchy_confusion_matrices(dataset_folder, hierarchy_csv)
    f1_score_species, f1_score_genus, f1_score_family, f1_score_order, f1_score_class = compute_hierarchy_perfs(hierarchy_conf_mat, encountered_species)

    return f1_score_species, f1_score_genus, f1_score_family, f1_score_order, f1_score_class

import csv

folder_list = ["torch_benchmark/models/baseline", "torch_benchmark/models/hierarchical_loss"]
export_csv = "torch_benchmark/results.csv"

# list models contained in folder_list
model_list = []
for folder in folder_list:
    model_list += [os.path.join(folder, model) for model in os.listdir(folder)]
    model_list.sort()

# if not present, create header (to csv
with open(export_csv, "w") as outfile:

            csv_writer = csv.DictWriter(outfile, fieldnames = ['model', 'run', 'Species_F1-score', 'Genus_F1-score', 'Family_F1-score', 'Order_F1-score', 'Class_F1-score'])
            csv_writer.writeheader()

# For each model, write results in a csv file
for model in model_list:
    model_name = model.split("/")[-1]
    model_name = model_name.split(".")[0] # remove .pth.tar
    print("Computing scores for: ", model_name)
    tmp = model_name.split("_")
    run_version = tmp[-1]
    model_name = "_".join(tmp[:-1])

    f1_score_species, f1_score_genus, f1_score_family, f1_score_order, f1_score_class = compute_hierarchy_perfs(model)
    # write values to csv:
    with open(export_csv, "a") as outfile:
        csv_writer = csv.DictWriter(outfile, fieldnames = ['model', 'run', 'Species_F1-score', 'Genus_F1-score', 'Family_F1-score', 'Order_F1-score', 'Class_F1-score'])
        csv_writer.writerow({'model': model_name,
                             'run': run_version,
                             'Species_F1-score': f1_score_species,
                             'Genus_F1-score': f1_score_genus,
                             'Family_F1-score': f1_score_family,
                             'Order_F1-score': f1_score_order,
                             'Class_F1-score': f1_score_class})


