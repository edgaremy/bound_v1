import os
import random
import shutil
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as patches

def split_dataset(dataset_dir, label_dir, train_dir, val_dir, test_dir, train_ratio, val_ratio):
    # Create the output directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(label_dir + 'train/', exist_ok=True)
    os.makedirs(label_dir + 'val/', exist_ok=True)
    os.makedirs(label_dir + 'test/', exist_ok=True)

    # Get the list of class folders in the dataset directory
    class_folders = [folder for folder in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, folder))]


    total_num_train = 0
    total_num_val = 0
    total_num_test = 0

    # Iterate over each class folder
    for class_folder in class_folders:
        class_path = os.path.join(dataset_dir, class_folder)
        files = os.listdir(class_path)
        random.shuffle(files)

        # Calculate the number of files for each split
        num_files = len(files)
        num_train = int(num_files * train_ratio)
        num_val = int(num_files * val_ratio)
        num_test = num_files - num_train - num_val
        print('Class: {}, Train: {}, Val: {}, Test: {}'.format(class_folder, num_train, num_val, num_test))
        total_num_test += num_test
        total_num_train += num_train
        total_num_val += num_val

        # Split the files into train, val, and test sets
        train_files = files[:num_train]
        val_files = files[num_train:num_train+num_val]
        test_files = files[num_train+num_val:]

        # Move the files to the corresponding split directories
        for file in train_files:
            src = os.path.join(class_path, file)
            # dst = os.path.join(train_dir, class_folder, file)
            dst = os.path.join(train_dir, file)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)
            label = file.split('.')[0] + '.txt'
            shutil.copy(os.path.join(label_dir, "default/", label), os.path.join(label_dir, "train/", label))

        for file in val_files:
            src = os.path.join(class_path, file)
            # dst = os.path.join(val_dir, class_folder, file)
            dst = os.path.join(val_dir, file)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)
            label = file.split('.')[0] + '.txt'
            shutil.copy(os.path.join(label_dir, "default/", label), os.path.join(label_dir, "val/", label))

        for file in test_files:
            src = os.path.join(class_path, file)
            # dst = os.path.join(test_dir, class_folder, file)
            dst = os.path.join(test_dir, file)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)
            label = file.split('.')[0] + '.txt'
            shutil.copy(os.path.join(label_dir, "default/", label), os.path.join(label_dir, "test/", label))

    print('Total number of train images: {}'.format(total_num_train))
    print('Total number of val images: {}'.format(total_num_val))
    print('Total number of test images: {}'.format(total_num_test))

# Example usage
dataset_dir = '/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD_71_visited/classes/'
label_dir = '/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD_71_visited/labels/'
train_dir = '/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD_71_visited/images/train/'
val_dir = '/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD_71_visited/images/val/'
test_dir = '/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD_71_visited/images/test/'
train_ratio = 0.8
val_ratio = 0.1
# test_ratio equals (1 - train_ratio - val_ratio)

split_dataset(dataset_dir, label_dir, train_dir, val_dir, test_dir, train_ratio, val_ratio)





# ### DISPLAY FULLSIZE IMAGE AS EXAMPLE ###

# # open img with corresponding label:
# imname = '2019-08-01 Luminy Hymenoptera 10 a 12 mm Apidae Amegilla 5'
# im = Image.open(f'/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD_71_visited/classes/Amegilla quadrifasciata/{imname}.jpeg')
# df = pd.read_csv(f'/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD_71_visited/labels/default/{imname}.txt', sep=' ', names=['class', 'x1', 'y1', 'w', 'h'])
# imr = np.array(im, dtype=np.uint8)

# # rescale coordinates:
# df_scaled = df.iloc[:, 1:]
# df_scaled[['x1', 'w']] = df_scaled[['x1', 'w']] * imr.shape[1]
# df_scaled[['y1', 'h']] = df_scaled[['y1', 'h']] * imr.shape[0]


# # display labels on image:
# fig,ax = plt.subplots(1, figsize=(10,10))# Display the image
# ax.imshow(imr)
# for box in df_scaled.values:
#     # Create a Rectangle patch
#     rect = patches.Rectangle((box[0]-(box[2]/2),box[1]-(box[3]/2)),box[2],box[3],linewidth=2,edgecolor='g',facecolor='none')# Add the patch to the axes
#     ax.add_patch(rect)
# plt.show()