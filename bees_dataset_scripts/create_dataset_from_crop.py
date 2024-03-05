import os
import random
import shutil

def split_dataset(labels_input_folder, labels_output_folder, image_input_folder, image_output_folder, train_val_ratio):
    # Create the output directories if they don't exist
    os.makedirs(os.path.join(image_output_folder,'train'), exist_ok=True)
    os.makedirs(os.path.join(image_output_folder,'val'), exist_ok=True)
    os.makedirs(os.path.join(image_output_folder,'test'), exist_ok=True)
    os.makedirs(os.path.join(labels_output_folder,'train'), exist_ok=True)
    os.makedirs(os.path.join(labels_output_folder,'val'), exist_ok=True)
    os.makedirs(os.path.join(labels_output_folder,'test'), exist_ok=True)

    # Calculate number of files for train/val split
    files = os.listdir(image_input_folder)
    random.shuffle(files)
    num_files = len(files)
    num_train = int(num_files * train_ratio)
    num_val = int(num_files * val_ratio)
    # num_test = int(num_files * test_ratio) # num_test = num_files - num_train - num_val
    print('Train: {}, Val: {}, Test: {}'.format(num_train, num_val, 0))

    # Split the files into train, val, and test sets
    train_files = files[:num_train]
    val_files = files[num_train:]
    # val_files = files[num_train:num_train+num_val]
    # test_files = files[num_train+num_val:]

    for image in train_files:
        image_name = os.path.splitext(image)[0]
        image_path = os.path.join(image_input_folder, image)
        txt_path = os.path.join(labels_input_folder, image_name + '.txt')
        shutil.copy(image_path, os.path.join(image_output_folder, 'train', image))
        if os.path.exists(txt_path):
            shutil.copy(txt_path, os.path.join(labels_output_folder, 'train', image_name + '.txt'))

    for image in val_files:
        image_name = os.path.splitext(image)[0]
        image_path = os.path.join(image_input_folder, image)
        txt_path = os.path.join(labels_input_folder, image_name + '.txt')
        shutil.copy(image_path, os.path.join(image_output_folder, 'val', image))
        if os.path.exists(txt_path):
            shutil.copy(txt_path, os.path.join(labels_output_folder, 'val', image_name + '.txt'))
    
    # for image in test_files:
    #     image_name = os.path.splitext(image)[0]
    #     image_path = os.path.join(image_input_folder, image)
    #     txt_path = os.path.join(labels_input_folder, image_name + '.txt')
    #     shutil.copy(image_path, os.path.join(image_output_folder, 'test',image))
    #     if os.path.exists(txt_path):
    #         shutil.copy(txt_path, os.path.join(labels_output_folder, 'test', image_name + '.txt'))

    print('Total number of train images: {}'.format(num_train))
    print('Total number of val images: {}'.format(num_val))
    # print('Total number of test images: {}'.format(num_test))

# Example usage
# labels_input_folder = '/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT5/dataset(prediction)/labels_validated'
# labels_output_folder = '/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT5/dataset/labels'
# image_input_folder = '/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT5/Pictures'
# image_output_folder = '/mnt/disk1/datasets/iNaturalist/Arthropods/LIMIT5/dataset/images'
# train_ratio = 0.9
# val_ratio = 0.1
# test_ratio = 0.0 # test_ratio equals (1 - train_ratio - val_ratio)

# split_dataset(labels_input_folder, labels_output_folder, image_input_folder, image_output_folder, train_ratio, val_ratio)