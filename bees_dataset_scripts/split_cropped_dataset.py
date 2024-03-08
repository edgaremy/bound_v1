import os
import random
import shutil

def split_dataset(dataset_dir, testset_dir, output_dir, train_val_ratio):
    # Create the output directories if they don't exist
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get the list of class folders in the dataset directory
    class_folders = [folder for folder in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, folder))]
    test_class_folders = [folder for folder in os.listdir(testset_dir) if os.path.isdir(os.path.join(testset_dir, folder))]


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
        num_train = int(num_files * train_val_ratio)
        num_val = num_files - num_train
        print('Class: {}, Train: {}, Val: {}'.format(class_folder, num_train, num_val))
        total_num_train += num_train
        total_num_val += num_val

        # Split the files into train, val, and test sets
        train_files = files[:num_train]
        val_files = files[num_train:]

        # Move the files to the corresponding split directories
        for file in train_files:
            src = os.path.join(class_path, file)
            dst = os.path.join(train_dir, class_folder, file)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)

        for file in val_files:
            src = os.path.join(class_path, file)
            dst = os.path.join(val_dir, class_folder, file)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)

    for class_folder in test_class_folders:
        class_path = os.path.join(testset_dir, class_folder)
        test_files = os.listdir(class_path)
        num_test = len(test_files)
        total_num_test += num_test
        print('Class: {}, Test: {}'.format(class_folder, num_test))
        for file in test_files:
            src = os.path.join(class_path, file)
            dst = os.path.join(test_dir, class_folder, file)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)

    print('Total number of train images: {}'.format(total_num_train))
    print('Total number of val images: {}'.format(total_num_val))
    print('Total number of test images: {}'.format(total_num_test))

# Example usage
dataset_dir = '/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/BD_307_cropped/ALL_else'
testset_dir = '/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/BD_307_cropped/DG'
output_dir = '/mnt/disk1/datasets/Projet_Bees_Detection_Basile/data_bees_detection/BD307/BD_307_cropped/dataset'
train_val_ratio = 0.9
split_dataset(dataset_dir, testset_dir, output_dir, train_val_ratio)