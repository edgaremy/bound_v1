import os
import yaml

def get_classes_from_filenames(input_directory):
    classes = set()
    for filename in os.listdir(input_directory):
        if "_" in filename:
            class_name = filename.split("_")[0]
            classes.add(class_name)
    return list(classes)

# Example usage:
# input_directory = "/mnt/disk1/datasets/iNaturalist/Arthropods/dataset17/images/test"
# list_of_families = get_classes_from_filenames(input_directory)
# print(len(list_of_families))

def get_files_by_name(base_directory, name):
    subsets = ['train', 'val', 'test']
    files = {subset: [] for subset in subsets}
    
    for subset in subsets:
        subset_dir = os.path.join(base_directory, subset)
        if os.path.exists(subset_dir):
            for filename in os.listdir(subset_dir):
                if filename.startswith(name + "_"):
                    files[subset].append(os.path.join(subset_dir, filename))
    
    return files['train'], files['val'], files['test']

# Example usage:
# base_directory = "/mnt/disk1/datasets/iNaturalist/Arthropods/dataset17/images"
# name = "326614"
# train_files, val_files, test_files = get_files_by_name(base_directory, name)
# print(train_files)

def write_yolov8_dataset_yaml(name, yaml_filename, dataset_root):
    dataset_images = os.path.join(dataset_root, 'images')
    train_files, val_files, test_files = get_files_by_name(dataset_images, name)
    yaml_filename = os.path.join(dataset_root, name + '_subset.yaml')

    train_file = os.path.join(dataset_root, "subsets", name + '_train.txt')
    val_file = os.path.join(dataset_root, "subsets", name + '_val.txt')
    test_file = os.path.join(dataset_root, "subsets", name + '_test.txt')

    # Create subsets directory
    os.makedirs(os.path.join(dataset_root, "subsets"), exist_ok=True)
    # Write train, val, test files to txt
    with open(train_file, 'w') as file:
        for f in train_files:
            file.write(f + '\n')
    with open(val_file, 'w') as file:
        for f in val_files:
            file.write(f + '\n')
    with open(test_file, 'w') as file:
        for f in test_files:
            file.write(f + '\n')
    # keep on relative path
    train_file = os.path.relpath(train_file, dataset_root)
    val_file = os.path.relpath(val_file, dataset_root)
    test_file = os.path.relpath(test_file, dataset_root)
    
    data = {
        'path': dataset_root,
        'train': train_file,
        'val': val_file,
        'test': test_file,
        'nc': 1,
        'names': {0: '"Arthropod"'}
    }
    
    with open(yaml_filename, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)

# Example usage:
# yaml_filename = "/mnt/disk1/datasets/iNaturalist/Arthropods/dataset17/326614_subset.yaml"
# dataset_path = "/mnt/disk1/datasets/iNaturalist/Arthropods/dataset17/"
# name = "326614"
# write_yolov8_dataset_yaml(name, yaml_filename, dataset_path)


def write_datasets_for_each_class(input_directory, dataset_root):
    classes = get_classes_from_filenames(input_directory)
    for class_name in classes:
        yaml_filename = os.path.join(dataset_root, class_name + '_subset.yaml')
        write_yolov8_dataset_yaml(class_name, yaml_filename, dataset_root)

# Example usage:
input_directory = "/mnt/disk1/datasets/iNaturalist/Arthropods/dataset17/images/test"
dataset_path = "/mnt/disk1/datasets/iNaturalist/Arthropods/dataset17/"
write_datasets_for_each_class(input_directory, dataset_path)